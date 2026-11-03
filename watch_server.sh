#!/bin/bash
# watch_server.sh —— 操作台看门狗: 每 10 秒检查, 挂了自动拉起
KEY=$(cat ~/FinRL_DeepSeek/.web_key 2>/dev/null)
while true; do
  if ! curl -s -o /dev/null --max-time 3 "http://localhost:8000/api/status?key=$KEY"; then
    pkill -f 'server.py$' 2>/dev/null
    sleep 1
    cd ~/FinRL_DeepSeek/webapp
    nohup ~/finrl_env/bin/python server.py > server.log 2>&1 &
    echo "$(date '+%F %T') watchdog 重启了服务器" >> watchdog.log
  fi
  sleep 10
done
