Natural Language Processing with Disaster Tweets
非構造化データのテキストの分類

モデルはkerasを用いて実装しました. モデルの構築でEmbedding層を導入しました。
事前学習済みのGloVeコーパスモデルを使用して単語をベクトル表現しました. 50D, 100D, および200D, の3種類がありますが, 今回は100Dで実装しました。

socoreは0.79190でした. word2vecや単にOnehotで実装してscoreが向上するか時間がある時に試してみようと思います。
