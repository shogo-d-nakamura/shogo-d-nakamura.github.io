#!/bin/zsh

# ログファイルの定義（任意でログを残す）
logfile="./jekyll_viewer.log"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> $logfile

# Jekyll サーバーをバックグラウンドで起動
echo "🚀 Starting Jekyll server..." | tee -a $logfile
bundle exec jekyll s --livereload >> $logfile 2>&1 &
jekyll_pid=$!

# Jekyll が正常に起動しているかを確認（1秒ごとに最大10秒待つ）
success=false
for i in {1..10}; do
  if lsof -i :4000 >/dev/null 2>&1; then
    success=true
    break
  fi
  sleep 1
done

if ! $success; then
  echo "❌ Jekyll server did not start within 10 seconds." | tee -a $logfile
  kill $jekyll_pid >/dev/null 2>&1
  exit 1
fi

# ブラウザで開く
echo "🌐 Opening http://127.0.0.1:4000 in your default browser..." | tee -a $logfile
open http://127.0.0.1:4000

# スクリプト終了時にサーバーを停止
trap "echo '🛑 Stopping Jekyll server...' | tee -a $logfile; kill $jekyll_pid" EXIT

# サーバーが動いている間は待機
wait $jekyll_pid


