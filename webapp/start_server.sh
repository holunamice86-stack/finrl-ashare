#!/bin/bash
# 启动操作台服务并等待就绪（自动附带看门狗）
KEY=$(cat ~/FinRL_DeepSeek/.web_key 2>/dev/null)
if ! curl -s -o /dev/null --max-time 3 "http://localhost:8000/api/status?key=$KEY"; then
  cd ~/FinRL_DeepSeek/webapp
  nohup ~/finrl_env/bin/python server.py > server.log 2>&1 &
fi
# 确保看门狗在跑（每10秒自检, 挂了自动拉起）
if ! pgrep -f 'watch_server.sh' > /dev/null 2>&1; then
  nohup bash ~/FinRL_DeepSeek/webapp/watch_server.sh >/dev/null 2>&1 &
fi
for i in $(seq 1 30); do
  sleep 2
  if curl -s -o /dev/null --max-time 3 "http://localhost:8000/api/status?key=$KEY"; then
    echo "READY"
    exit 0
  fi
done
echo "FAIL - server did not start, see ~/FinRL_DeepSeek/webapp/server.log"
exit 1
