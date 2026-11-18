# 精简版 xiaogpt 自建镜像（792MB，官方 2.76GB）
# 所有补丁已固化进源码：Agent Loop / 动态日期 / 反幻觉提示词 / 请稍等
# 注意：oect 上 docker build 会卡在 pip 层，实际用手动容器法构建（见方案文档），
#       本 Dockerfile 仅作构建蓝图留存。
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Shanghai \
    XIAOGPT_MI_TOKEN_PATH=/config/.mi.token XDG_CONFIG_HOME=/config XIAOGPT_PORT=9527
WORKDIR /app
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
    "aiohttp>=3.14.4" "httpx2[socks]>=2.16.0" "httpx>=0.28" \
    "republic>=0.6.1" "rich>=15.0.0" "pyyaml>=6.0.3" "lingua-language-detector>=2.2.0"
COPY MiService-main /tmp/MiService-main
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --no-deps /tmp/MiService-main \
    && rm -rf /tmp/MiService-main
COPY xiaogpt-main/xiaogpt /app/xiaogpt
COPY xiaogpt-main/xiaogpt.py /app/xiaogpt.py
COPY xiaogpt-main/xiaogpt/agent_bot.py /app/agent_bot.py
VOLUME /config
EXPOSE 9527
ENTRYPOINT ["python", "-m", "xiaogpt.cli"]
