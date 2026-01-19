#!/bin/zsh

# ログファイルの定義（任意でログを残す）
logfile="./jekyll_viewer.log"

# ヘルプメッセージを表示
show_help() {
  cat << 'EOF'
Usage: zsh viewer.sh [OPTIONS]

Jekyll development server launcher with automatic browser opening.

OPTIONS:
  -h, --help     Show this help message and exit
  -s, --stop     Stop any running Jekyll servers and exit
  -k, --kill     Same as --stop (alias)

DESCRIPTION:
  This script starts a Jekyll development server with live reload enabled.
  It automatically:
    - Stops any existing Jekyll servers on port 4000
    - Starts a new Jekyll server in the background
    - Opens http://127.0.0.1:4000 in your default browser
    - Logs output to ./jekyll_viewer.log

  Press Ctrl+C to stop the server when running.

EXAMPLES:
  zsh viewer.sh          # Start the Jekyll server
  zsh viewer.sh --help   # Show this help
  zsh viewer.sh --stop   # Stop running Jekyll servers

EOF
  exit 0
}

# 既存のサーバーを停止する関数
stop_servers() {
  echo "🔍 Checking for running Jekyll servers..."

  local found=false

  # ポート 4000 を使用しているプロセスを検索
  existing_pids=$(lsof -t -i :4000 2>/dev/null)
  if [[ -n "$existing_pids" ]]; then
    found=true
    echo "⚠️  Found process(es) on port 4000: $existing_pids"
    echo "🛑 Stopping..."
    for pid in ${(f)existing_pids}; do
      kill -TERM $pid 2>/dev/null
    done
    sleep 2
    # まだ残っている場合は強制終了
    existing_pids=$(lsof -t -i :4000 2>/dev/null)
    if [[ -n "$existing_pids" ]]; then
      echo "⚠️  Force killing remaining process(es): $existing_pids"
      for pid in ${(f)existing_pids}; do
        kill -9 $pid 2>/dev/null
      done
      sleep 1
    fi
  fi

  # jekyll プロセス名でも検索
  jekyll_procs=$(pgrep -f "jekyll.*serve" 2>/dev/null)
  if [[ -n "$jekyll_procs" ]]; then
    found=true
    echo "⚠️  Found Jekyll processes: $jekyll_procs"
    echo "🛑 Stopping..."
    for pid in ${(f)jekyll_procs}; do
      kill -TERM $pid 2>/dev/null
    done
    sleep 1
  fi

  if $found; then
    echo "✅ All Jekyll servers stopped."
  else
    echo "✅ No running Jekyll servers found."
  fi
}

# 引数の処理
case "$1" in
  -h|--help)
    show_help
    ;;
  -s|--stop|-k|--kill)
    stop_servers
    exit 0
    ;;
esac

echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> $logfile

# 既存の Jekyll サーバーを検索して停止
echo "🔍 Checking for existing Jekyll servers..." | tee -a $logfile

# ポート 4000 を使用しているプロセスを検索
existing_pids=$(lsof -t -i :4000 2>/dev/null)
if [[ -n "$existing_pids" ]]; then
  echo "⚠️  Found existing process(es) on port 4000: $existing_pids" | tee -a $logfile
  echo "🛑 Stopping existing Jekyll server(s)..." | tee -a $logfile
  for pid in ${(f)existing_pids}; do
    kill -TERM $pid 2>/dev/null
  done
  # プロセスが終了するまで少し待つ
  sleep 2
  # まだ残っている場合は強制終了
  existing_pids=$(lsof -t -i :4000 2>/dev/null)
  if [[ -n "$existing_pids" ]]; then
    echo "⚠️  Force killing remaining process(es): $existing_pids" | tee -a $logfile
    for pid in ${(f)existing_pids}; do
      kill -9 $pid 2>/dev/null
    done
    sleep 1
  fi
  echo "✅ Existing server(s) stopped." | tee -a $logfile
fi

# jekyll プロセス名でも検索して停止（念のため）
jekyll_procs=$(pgrep -f "jekyll.*serve" 2>/dev/null)
if [[ -n "$jekyll_procs" ]]; then
  echo "⚠️  Found Jekyll processes by name: $jekyll_procs" | tee -a $logfile
  for pid in ${(f)jekyll_procs}; do
    kill -TERM $pid 2>/dev/null
  done
  sleep 1
fi

# Jekyll サーバーをバックグラウンドで起動
echo "🚀 Starting Jekyll server..." | tee -a $logfile
bundle exec jekyll s --livereload >> $logfile 2>&1 &
jekyll_pid=$!

# PID を記録
echo "📝 Jekyll PID: $jekyll_pid" | tee -a $logfile

# クリーンアップ関数
cleanup() {
  echo "" | tee -a $logfile
  echo "🛑 Stopping Jekyll server (PID: $jekyll_pid)..." | tee -a $logfile

  # まず TERM シグナルで停止を試みる
  if kill -0 $jekyll_pid 2>/dev/null; then
    kill -TERM $jekyll_pid 2>/dev/null

    # 最大5秒待つ
    for i in {1..5}; do
      if ! kill -0 $jekyll_pid 2>/dev/null; then
        echo "✅ Jekyll server stopped gracefully." | tee -a $logfile
        break
      fi
      sleep 1
    done

    # まだ動いている場合は強制終了
    if kill -0 $jekyll_pid 2>/dev/null; then
      echo "⚠️  Force killing Jekyll server..." | tee -a $logfile
      kill -9 $jekyll_pid 2>/dev/null
      sleep 1
    fi
  fi

  # ポート 4000 に残っているプロセスも確認して停止
  remaining=$(lsof -t -i :4000 2>/dev/null)
  if [[ -n "$remaining" ]]; then
    echo "⚠️  Cleaning up remaining processes on port 4000: $remaining" | tee -a $logfile
    for pid in ${(f)remaining}; do
      kill -9 $pid 2>/dev/null
    done
  fi

  echo "✅ Cleanup complete." | tee -a $logfile
  exit 0
}

# 複数のシグナルに対応したトラップを設定
trap cleanup INT TERM HUP QUIT EXIT

# Jekyll が正常に起動しているかを確認（1秒ごとに最大15秒待つ）
success=false
for i in {1..15}; do
  # プロセスがまだ存在するか確認
  if ! kill -0 $jekyll_pid 2>/dev/null; then
    echo "❌ Jekyll process died unexpectedly." | tee -a $logfile
    exit 1
  fi

  if lsof -i :4000 >/dev/null 2>&1; then
    success=true
    break
  fi
  echo "⏳ Waiting for server to start... ($i/15)" | tee -a $logfile
  sleep 1
done

if ! $success; then
  echo "❌ Jekyll server did not start within 15 seconds." | tee -a $logfile
  kill $jekyll_pid 2>/dev/null
  exit 1
fi

echo "✅ Jekyll server started successfully!" | tee -a $logfile

# ブラウザで開く
echo "🌐 Opening http://127.0.0.1:4000 in your default browser..." | tee -a $logfile
open http://127.0.0.1:4000

echo "💡 Press Ctrl+C to stop the server." | tee -a $logfile

# サーバーが動いている間は待機
wait $jekyll_pid
