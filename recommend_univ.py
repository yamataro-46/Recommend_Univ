# 大学学部提案システム

# 初期準備
# import, リスト・辞書の定義
from janome.tokenizer import Tokenizer
from gensim import models
import streamlit as st
import time


# ファイルの読み込み

# 白ヤギモデル・大学学部レビューモデルの融合学習モデルの読み込み
fus_model = models.Word2Vec.load('gensim_univs.model')

# 学習させる文書をレビュー全文の分かち書きとしたdoc2vecモデル
wakati_model = models.Doc2Vec.load('wakati_reviews.model')



# 関数等定義

# 入力文を分かち書きする
def wakati_sentence(text):
    tw = Tokenizer(wakati=True)
    token = tw.tokenize(text)
    tokenList = list(token)
    return tokenList


# 文章を形態素解析(分かち書き)する　単語群と出現回数リストを返す関数
def wakati_count_words(contents):
    t1 = Tokenizer()
    words = []
    words_count = {}
    for token in t1.tokenize(contents):
        parts = token.part_of_speech.split(',')
        if('名詞' in parts
            or '動詞' in parts
            or '形容詞' in parts
            or '形容動詞' in parts):
            base = token.base_form
            words.append(base)
            rec = words_count.get(base)
            if rec == None:
                words_count[base] = 1
            else:
                words_count[base] = rec + 1
    return words, words_count


# 白ヤギ+レビューのword2vecモデルを用いて、単語の似ているワードをリスト化
def similar_words(text, model, n):
    sentence = wakati_sentence(text)
    words, words_count = wakati_count_words(text)
    wordsList = sentence
    for word in words:
        sim_words = model.wv.most_similar(word, topn=n)
        for w in range(len(sim_words)):
            wordsList.append(sim_words[w][0])
    return wordsList



# 入力文から、最も近い・合っている大学学部を出力(提案)する
def output_similar_univ(input_text, model, n):
    text = similar_words(input_text, fus_model, 10)

    # 新規文書のモデルにおけるベクトルを作成する
    input_vec = model.infer_vector(text)
    
    # 新規文書のベクトルと近いものを出力 ⇒ 条件や要件に近い・合っている大学学部を提案する
    output_univs = model.docvecs.most_similar([input_vec], topn=n)

    return output_univs




# webアプリとして公開
st.title('あなたに合った大学学部を見つけよう!')

'あなたがどんな大学に通いたいか、どんな大学生活を送りたいか 希望を書いてみましょう。'
'きっとあなたに合う大学学部を教えてくれますよ'


input_text = st.text_input('大学学部を選ぶ上での条件')


if input_text != '':
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(f'検索中... {i+1}')
        bar.progress(i+1)
        time.sleep(0.05)

    '「', input_text, '」 という条件に近い大学学部はこちら'
    ans = output_similar_univ(input_text, wakati_model, 30)
    # 上位5位を表示
    for i in range(5):
        '第' , i+1, '位：', ans[i][0], '　条件一致度：', round(float(ans[i][1])*100, 2), '%'
    expander = st.expander('6位以降はこちらから')
    for i in range(5, len(ans)):
        ex = '第' + str(i+1) + '位：' + ans[i][0] + '　条件一致度：' + str(round(float(ans[i][1])*100, 2)) + '%'
        expander.write(ex)
