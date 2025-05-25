---
layout: post
title: MCPでAutoDock Vinaを動かす
date: 2025-05-25 23:09 +0900
description: ''
category: ''
tags: [MCP, Docking, Python, LLM, Claude]
published: true
---


# Model Concept Protocol (MCP)
参考1: [zenn](https://zenn.dev/cloud_ace/articles/model-context-protocol) \\
参考2: [DeepLearning.ai](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/) \
MCPクライアント(Claude DesktopとかCursorとか)からMCPサーバーにリクエストを投げたら、MCPサーバーが持っている情報にアクセスして検索結果を返してくれます。LLMが外部の情報にアクセスできるようになるのが良い点です。


# Claude Desktop からSMILESと標的名を投げてドッキング計算
とりあえず触ってみるためにvinaで計算したドッキングスコアを計算してもらおうと思います。MCPサーバーのリソースには標的タンパク質の名前とconfigの情報(ポケット中心とかグリッドサイズとか)だけ与えておき、Claude DesktopでSMILESと事前に定義した標的名を指定したらドッキングが走るようにします。


ソースコード:\
[https://github.com/shogo-d-nakamura/MCP_Vina](https://github.com/shogo-d-nakamura/MCP_Vina)



参考にしたもの: \
[Claude document](https://modelcontextprotocol.io/quickstart/server) \
[chatMol](https://github.com/ytworks/chatMol) \
[ブログ](https://iwatobipen.wordpress.com/2025/05/04/integration-chembl-rest-api-and-claude-with-mcp-cheminformatics-mcp-ai/)


## Claude Desktopのダウンロード
MCPのサーバーのたてかたについて、Claudeからドキュメントが公開されています(↑のリンク)。これに沿ってやるのでClaude Desktopを使えるようにしておきます。後に出てくるconfigのjsonをを移植するだけでCursorとかでも動きます。


## uvで仮想環境作成

```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

curlの後にrestart terminalするとuvが使えるようになります。 \
公開したリポジトリにpyproject.tomlがあるので、
```zsh
uv pip install -e .
```
で一通りパッケージがインストールされます。


自分で一からパッケージを管理する場合はClaudeのドキュメントにもあるように`uv init`した後に`uv add`で必要なものを追加していきます。
uv init $DIR_NAME -p 3.10 でpythonのバージョンを指定できます。

```zsh
uv init test -p 3.10
cd test
source .venv/bin/activate
# install packages
uv add httpx rdkit meeko "mcp[cli]"
```


SMILESから３次元構造を生成するのにscrubber(現molscrub)を使いました。scrubberのscrub.pyと、meekoのmk_prepare_ligand.pyのPATHが通っていればリガンドの前処理ができます。
```bash
git clone --single-branch --branch develop https://github.com/forlilab/scrubber.git
cd scrubber; uv pip install -e .; cd .. 
```


## Claude Desktop の設定
Claude DesktopのSettings -> Developer -> Edit Conifg で、jsonファイルが開きます。ここに、以下をコピペします。
![Settings](/assets/img/2025_images/claude_settings.png)

```json
{
  "mcpServers": {
    "molecular-docking": {
      "command": "uv",
      "args": [
        "--directory",
        "/PATH/TO/molDocking",
        "run",
        "server.py"
      ]
    }
  },
  "globalShortcut": ""
}
```
Settingsを更新したら、Claudeを再起動します。

Claudeのドキュメントはこれで動いてるっぽいですが、自分の場合はここでエラーが出ました(macbook m4, Sequoia)。uvが認識されていない感じだったので、which uvで取得した絶対パスを入力すると正常に認識されました。これを別のクライアントのMCP configのjsonにコピペするだけで同じように接続できます。標準化サイコゥーです。

```json
{
  "mcpServers": {
    "molecular-docking": {
      "command": "/PATH/TO/uv",
      "args": [
        "--directory",
        "/PATH/TO/molDocking",
        "run",
        "server.py"
      ]
    }
  },
  "globalShortcut": ""
}
```

## 使ってみる

### 標的タンパク質の種類

標的タンパク質に何が登録されているか聞いてみると、AKT1が返ってきました。これは、/molDocking/server.pyのmcp.tool()デコレータが付いたlist_available_targetsという関数を動かしてくれています。あらかじめAKT1のpdbqtとconfigの情報だけ置いてあるので、それを返してくれています。

```python
@mcp.tool()
def list_available_targets() -> Dict[str, Any]:
~
```
![](/assets/img/2025_images/dock1.png)

### config.txtの情報
ポケット中心の座標とかを出してくれる関数も@mcp.tool()デコレータを付けて書いてあります。
![](/assets/img/2025_images/dock2.png)

### 実際に計算を投げる
適当なSMILESをAKT1にドッキングしてもらいます。@mcp.tool()のrun_molecular_docking()が実行されます。
![](/assets/img/2025_images/dock3.png)
役に立ちませんがClaudeなりの解釈も述べてくれます。
![](/assets/img/2025_images/dock4.png)



以上で、あらかじめ登録した標的タンパク質の名前とSMILESを投げたらVinaのスコアが返ってくるようになりました。今回はMCPにタンパク質のpdbqtとconfigを置いただけですが、ローカルのファイルやREST APIなど色々組み合わせればLLMをラッパーとして分子デザイン的なことができそうです。