import spacy
import ginza
import numpy as np
        
class FeatureExtractor:
    def __init__(self, text):
        self.text = text
        nlp = spacy.load('ja_ginza')
        self.doc = nlp(text)
        self.vector = {}
    
    # ベクトル化
    def extract_future_vector(self) -> np.ndarray:
        hinshi_bigram_vec = self._extract_hinshi_bigrams()
        zyoshi_bigram_vec = self._extract_zyoshi_bigrams()
        kakariuke_vec = self._extract_kakariuke()
        kutouten_vec = self._extract_kutouten()
        kinougo_vec = self._extract_kinougo()
        syouryaku_vec = self._extract_syouryaku()
        return np.concatenate([hinshi_bigram_vec, zyoshi_bigram_vec, kakariuke_vec, kutouten_vec, kinougo_vec, syouryaku_vec])
    
    # 品詞のバイグラム
    def _extract_hinshi_bigrams(self) -> np.ndarray:
        bigrams = np.zeros((34, 34)) # 34*34の行列
        for sent in self.doc.sents:
            for i in range(len(sent) - 1):
                tag1 = self._get_hinshi_tag(sent[i].tag_)
                tag2 = self._get_hinshi_tag(sent[i + 1].tag_)
                bigrams[tag1, tag2] += 1
        return bigrams.ravel() # 1次元ベクトルに変換
     
    def _get_hinshi_tag(self, tag) -> int:
        # 品詞の分類
        tags = ["名詞-普通名詞", "名詞-固有名詞", "名詞-数詞", "名詞-助動詞語幹", "代名詞", "形状詞-一般", "形状詞-タリ", "形状詞-助動詞語幹", "連体詞", "副詞", "接続詞", "感動詞-一般", "感動詞-フィラー", "動詞-一般", "動詞-非自立可能", "形容詞-一般", "形容詞-非自立可能", "助動詞", "助詞-格助詞", "助詞-副助詞", "助詞-係助詞", "助詞-接続助詞", "助詞-終助詞", "助詞-準体助詞", "接頭辞", "接尾辞-名詞的", "接尾辞-形状詞的", "接尾辞-動詞的", "接尾辞-形容詞的", "記号", "補助記号-句点", "補助記号-読点", "補助記号-括弧",  "空白"]
        # GiNZAから出力される記号のタグを統合
        if tag.startswith("記号"):
            tag = "記号"
        elif tag.startswith("補助記号"):
            if tag == "補助記号-句点":
                tag = "補助記号-句点"
            elif tag == "補助記号-読点":
                tag = "補助記号-読点"
            elif tag.stawrtswith("補助記号-括弧"):
                tag = "補助記号-括弧"
            else:
                tag = "記号"
        tag = "-".join(tag.split('-')[:2]) # ハイフンで3つ目以降の分類は無視
        return tags.index(tag)
    
    # 助詞のバイグラム
    def _extract_zyoshi_bigrams(self) -> np.ndarray:
        # 助詞の一覧
        zyoshi_list = {
            "助詞-格助詞": ["の", "に", "は", "を", "が", "と", "で", "から", "へ", "より"],
            "助詞-副助詞": ["か", "や", "など", "だけ", "って", "たり", "まで"],
            "助詞-係助詞": ["は", "も"],
            "助詞-接続助詞": ["て", "が", "と", "ば", "から", "けれど", "ながら"],
            "助詞-終助詞": ["か", "ね", "よ", "な"],
            "助詞-準体助詞": ["の"]
        }
        words = [f"{t}-{w}" for t, ws in zyoshi_list.items() for w in ws] # 31
        bigrams = np.zeros((31, 31))
        for sent in self.doc.sents:
            # リストに含まれる助詞のみに絞る
            zyoshi_sent = [words.index(f"{token.tag_}-{token.text}") for token in sent if f"{token.tag_}-{token.text}" in words]
            for i in range(len(zyoshi_sent) - 1):
                bigrams[zyoshi_sent[i], zyoshi_sent[i + 1]] += 1
        return bigrams.ravel()
    
    # 係り受けの特徴
    def _extract_kakariuke(self) -> np.ndarray:
        heights = [] # 高さ
        depths = [] # 深さ
        distances = [] # 距離
        branches = [] # 分岐数
        for sent in self.doc.sents:
            part_depths = []
            for span in ginza.bunsetu_spans(sent):
                depth = len(list(ginza.bunsetu_span(span.root)))
                part_depths.append(depth)
                branches.append(len(list(span.lefts)))
                for token in span.lefts:
                    distance = abs(token.i - span.root.i)
                    distances.append(distance)
            depths.extend(part_depths)
            heights.append(max(part_depths))
        # それぞれの max/min/mean/std を計算
        kakariuke_vector = [
            np.max(heights), np.min(heights), np.mean(heights), np.std(heights),
            np.max(depths), np.min(depths), np.mean(depths), np.std(depths),
            np.max(distances), np.min(distances), np.mean(distances), np.std(distances),
            np.max(branches), np.min(branches), np.mean(branches), np.std(branches)
        ]
        return np.array(kakariuke_vector)
    
    # 句読点の分布
    def _extract_kutouten(self) -> np.ndarray:
        kutouten_counts = np.zeros(10) # 文章を10区間に分けてカウント
        # 句読点の分布を各文章ごとに計算
        for sent in self.doc.sents:
            length = len(sent)
            base_i = sent.start # 部分文の開始位置 -> 相対位置を長さで割って処理
            for token in sent:
                if token.is_punct: # 句読点
                    idx = int((token.i - base_i) / length * 10)
                    kutouten_counts[idx] += 1
        return kutouten_counts
    
    # 機能語の使用頻度
    def _extract_kinougo(self) -> np.ndarray:
        # 機能語の分類
        kinougo_tags = ["連体詞", "副詞", "接続詞", "助動詞", "助詞-格助詞", "助詞-副助詞", "助詞-係助詞", "助詞-接続助詞", "助詞-終助詞", "助詞-準体助詞"]
        # 機能語の一覧 (漢字とひらがなの表記揺れを考慮)
        kinougo_list = ["の-助詞-格助詞", "に-助詞-格助詞", "て-助詞-接続助詞", "は-助詞-係助詞", "を-助詞-格助詞", "が-助詞-格助詞", "と-助詞-格助詞", "で-助詞-格助詞", "も-助詞-係助詞", "の-助詞-準体助詞", "から-助詞-格助詞", "が-助詞-接続助詞", "か-助詞-終助詞", "其の-連体詞", "か-助詞-副助詞", "此の-連体詞", "と-助詞-接続助詞", "や-助詞-副助詞", "ば-助詞-接続助詞", "から-助詞-接続助詞", "など-助詞-副助詞", "ね-助詞-終助詞", "よ-助詞-終助詞", "まで-助詞-副助詞", "へ-助詞-格助詞", "そう-副詞", "だけ-助詞-副助詞", "どう-副詞", "又-接続詞", "って-助詞-副助詞", "けれど-助詞-接続助詞", "な-助詞-終助詞", "たり-助詞-副助詞", "ながら-助詞-接続助詞", "然し-接続詞", "より-助詞-格助詞", "し-助詞-接続助詞", "そして-接続詞", "もう-副詞", "こう-副詞", "ほど-助詞-副助詞", "くらい-助詞-副助詞", "及び-接続詞", "良く-副詞", "同じ-連体詞", "そんな-連体詞", "少し-副詞", "彼の-連体詞", "の-助詞-終助詞", "未だ-副詞", "矢張り-副詞", "しか-助詞-副助詞", "わ-助詞-終助詞", "大きな-連体詞", "一寸-副詞", "又-副詞", "先ず-副詞", "直ぐ-副詞", "特に-副詞", "ばかり-助詞-副助詞", "何の-連体詞", "或いは-接続詞", "例えば-副詞", "こんな-連体詞", "何故-副詞", "全く-副詞", "なんて-助詞-副助詞", "一番-副詞", "或る-連体詞", "勿論-副詞", "余り-副詞", "詰まり-副詞", "若し-副詞", "我が-連体詞", "既に-副詞", "更に-副詞", "で-接続詞", "どんな-連体詞", "初めて-副詞", "更に-接続詞", "迚も-副詞", "最も-副詞", "可成-副詞", "もっと-副詞", "のみ-助詞-副助詞", "こそ-助詞-係助詞", "唯-接続詞", "必ず-副詞", "然も-接続詞", "さ-助詞-終助詞", "まあ-副詞", "つつ-助詞-接続助詞", "色々-副詞", "小さな-連体詞", "さえ-助詞-副助詞", "ぞ-助詞-終助詞", "猶-接続詞", "ずっと-副詞", "中々-副詞", "但し-接続詞", "唯-副詞", "なんか-助詞-副助詞", "より-副詞", "即ち-接続詞", "沢山-副詞", "はっきり-副詞", "確り-副詞", "所謂-連体詞", "暫く-副詞", "とも-助詞-接続助詞", "ずつ-助詞-副助詞", "再び-副詞", "且つ-接続詞", "決して-副詞", "丸で-副詞", "ゆっくり-副詞", "一方-接続詞", "扠-接続詞", "寧ろ-副詞", "結構-副詞", "如何に-副詞", "多分-副詞", "正に-副詞", "極めて-副詞", "恐らく-副詞", "軈て-副詞", "兎に角-副詞", "一体-副詞", "略-副詞", "嘗て-副詞", "急度-副詞", "やら-助詞-副助詞", "直接-副詞", "ちゃんと-副詞", "是非-副詞", "宜しく-副詞", "い-助詞-終助詞", "突然-副詞", "若しくは-接続詞", "丁度-副詞", "きちんと-副詞", "十分-副詞", "大変-副詞", "色んな-連体詞", "当然-副詞", "元々-副詞", "全然-副詞", "にて-助詞-格助詞", "漸く-副詞", "あらゆる-連体詞", "一層-副詞", "漸と-副詞", "けれど-接続詞", "取り敢えず-副詞", "遂に-副詞", "やや-副詞", "どんどん-副詞", "が-接続詞", "流石-副詞", "しも-助詞-副助詞", "かしら-助詞-終助詞", "もの-助詞-終助詞", "すっかり-副詞", "改めて-副詞", "仮令-副詞", "随分-副詞", "単に-副詞", "ぜ-助詞-終助詞", "飽くまで-副詞", "益々-副詞", "じっと-副詞", "どうぞ-副詞", "すら-助詞-副助詞", "行成-副詞", "次々-副詞", "言わば-副詞", "未だ-副詞", "幾ら-副詞", "極く-副詞", "主な-連体詞", "どころ-助詞-副助詞", "つい-副詞", "一杯-副詞", "未だ未だ-副詞", "たって-助詞-接続助詞", "最早-副詞", "一旦-副詞", "たっぷり-副詞", "猶-副詞", "きり-助詞-副助詞", "同じく-副詞", "偶に-副詞", "屡-副詞", "僅か-副詞", "兎も角-副詞", "ふと-副詞", "愈-副詞", "尤も-接続詞", "主に-副詞", "折角-副詞"]
        kinougo_list_hiragana = ["の-助詞-格助詞", "に-助詞-格助詞", "て-助詞-接続助詞", "は-助詞-係助詞", "を-助詞-格助詞", "が-助詞-格助詞", "と-助詞-格助詞", "で-助詞-格助詞", "も-助詞-係助詞", "の-助詞-準体助詞", "から-助詞-格助詞", "が-助詞-接続助詞", "か-助詞-終助詞", "その-連体詞", "か-助詞-副助詞", "この-連体詞", "と-助詞-接続助詞", "や-助詞-副助詞", "ば-助詞-接続助詞", "から-助詞-接続助詞", "など-助詞-副助詞", "ね-助詞-終助詞", "よ-助詞-終助詞", "まで-助詞-副助詞", "へ-助詞-格助詞", "そう-副詞", "だけ-助詞-副助詞", "どう-副詞", "また-接続詞", "って-助詞-副助詞", "けれど-助詞-接続助詞", "な-助詞-終助詞", "たり-助詞-副助詞", "ながら-助詞-接続助詞", "しかし-接続詞", "より-助詞-格助詞", "し-助詞-接続助詞", "そして-接続詞", "もう-副詞", "こう-副詞", "ほど-助詞-副助詞", "くらい-助詞-副助詞", "および-接続詞", "よく-副詞", "おなじ-連体詞", "そんな-連体詞", "すこし-副詞", "あの-連体詞", "の-助詞-終助詞", "まだ-副詞", "やはり-副詞", "しか-助詞-副助詞", "わ-助詞-終助詞", "おおきな-連体詞", "ちょっと-副詞", "また-副詞", "まず-副詞", "すぐ-副詞", "とくに-副詞", "ばかり-助詞-副助詞", "どの-連体詞", "あるいは-接続詞", "たとえば-副詞", "こんな-連体詞", "なぜ-副詞", "まったく-副詞", "なんて-助詞-副助詞", "いちばん-副詞", "ある-連体詞", "もちろん-副詞", "あまり-副詞", "つまり-副詞", "もし-副詞", "わが-連体詞", "すでに-副詞", "さらに-副詞", "で-接続詞", "どんな-連体詞", "はじめて-副詞", "さらに-接続詞", "とても-副詞", "もっとも-副詞", "かなり-副詞", "もっと-副詞", "のみ-助詞-副助詞", "こそ-助詞-係助詞", "ただ-接続詞", "かならず-副詞", "しかも-接続詞", "さ-助詞-終助詞", "まあ-副詞", "つつ-助詞-接続助詞", "いろいろ-副詞", "ちいさな-連体詞", "さえ-助詞-副助詞", "ぞ-助詞-終助詞", "なお-接続詞", "ずっと-副詞", "なかなか-副詞", "ただし-接続詞", "ただ-副詞", "なんか-助詞-副助詞", "より-副詞", "すなわち-接続詞", "たくさん-副詞", "はっきり-副詞", "しっかり-副詞", "いわゆる-連体詞", "しばらく-副詞", "とも-助詞-接続助詞", "ずつ-助詞-副助詞", "ふたたび-副詞", "かつ-接続詞", "けっして-副詞", "まるで-副詞", "ゆっくり-副詞", "いっぽう-接続詞", "さて-接続詞", "むしろ-副詞", "けっこう-副詞", "いかに-副詞", "たぶん-副詞", "まさに-副詞", "きわめて-副詞", "おそらく-副詞", "やがて-副詞", "とにかく-副詞", "いったい-副詞", "ほぼ-副詞", "かつて-副詞", "きっと-副詞", "やら-助詞-副助詞", "ちょくせつ-副詞", "ちゃんと-副詞", "ぜひ-副詞", "よろしく-副詞", "い-助詞-終助詞", "とつぜん-副詞", "もしくは-接続詞", "ちょうど-副詞", "きちんと-副詞", "じゅうぶん-副詞", "たいへん-副詞", "いろんな-連体詞", "とうぜん-副詞", "もともと-副詞", "ぜんぜん-副詞", "にて-助詞-格助詞", "ようやく-副詞", "あらゆる-連体詞", "いっそう-副詞", "やっと-副詞", "けれど-接続詞", "とりあえず-副詞", "ついに-副詞", "やや-副詞", "どんどん-副詞", "が-接続詞", "さすが-副詞", "しも-助詞-副助詞", "かしら-助詞-終助詞", "もの-助詞-終助詞", "すっかり-副詞", "あらためて-副詞", "たとえ-副詞", "ずいぶん-副詞", "たんに-副詞", "ぜ-助詞-終助詞", "あくまで-副詞", "ますます-副詞", "じっと-副詞", "どうぞ-副詞", "すら-助詞-副助詞", "いきなり-副詞", "つぎつぎ-副詞", "いわば-副詞", "いまだ-副詞", "いくら-副詞", "ごく-副詞", "おもな-連体詞", "どころ-助詞-副助詞", "つい-副詞", "いっぱい-副詞", "まだまだ-副詞", "たって-助詞-接続助詞", "もはや-副詞", "いったん-副詞", "たっぷり-副詞", "なお-副詞", "きり-助詞-副助詞", "おなじく-副詞", "たまに-副詞", "しばしば-副詞", "わずか-副詞", "ともかく-副詞", "ふと-副詞", "いよいよ-副詞", "もっとも-接続詞", "おもに-副詞", "せっかく-副詞"]
        # 機能語の頻度
        kinougo_counts = np.zeros(200)
        for sent in self.doc.sents:
            for token in sent:
                if f"{token.lemma_}-{token.tag_}" in kinougo_list_hiragana:
                    idx = kinougo_list_hiragana.index(f"{token.lemma_}-{token.tag_}")
                    kinougo_counts[idx] += 1
                elif f"{token.lemma_}-{token.tag_}" in kinougo_list:
                    idx = kinougo_list.index(f"{token.text}-{token.tag_}")
                    kinougo_counts[idx] += 1
        return kinougo_counts
    
    # 省略傾向
    def _extract_syouryaku(self) -> np.ndarray:
        # 動詞の従属関係に主部(csubj)や目的部(obj)があるかを調べる
        verbs = [token for token in self.doc if token.pos_ == "VERB"]
        syouryaku_vector = np.zeros(2) # 主語と目的語の省略率
        for verb in verbs:
            if not any(child.dep_ == "nsubj" for child in verb.children):
                syouryaku_vector[0] += 1
            if not any(child.dep_ == "obj" for child in verb.children):
                syouryaku_vector[1] += 1
        syouryaku_vector /= len(verbs) if verbs else np.zeros(2)
        return syouryaku_vector
                
if __name__ == "__main__":
    # sample_text = "これがサンプルが作ったテキストで、NLPを試すために作られています。米カワウソペンギン犬猫狐が好きです。"
    sample_text = "これはサンプルテキストです。私が書きました。"
    extractor = FeatureExtractor(sample_text)
    r = extractor.extract_future_vector()
    print(r.shape)

    

           
                
            
    
    
    
    
