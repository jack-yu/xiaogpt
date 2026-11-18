# 精简镜像 —— 在 xiaogpt fork 仓库根目录构建（docker build -f Dockerfile.fork .）
# 基于 python:3.12-slim，砍掉上游用不到的第三方 TTS 大依赖，约 792MB（上游 2.76GB）。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Shanghai \
    XIAOGPT_MI_TOKEN_PATH=/config/.mi.token XDG_CONFIG_HOME=/config

WORKDIR /app

# 只装实际需要的依赖（不含 tetos / azure / google-cloud / edge-tts 等）
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
    "aiohttp>=3.14.4" "httpx2[socks]>=2.16.0" "httpx>=0.28" \
    "republic>=0.6.1" "rich>=15.0.0" "pyyaml>=6.0.3" "lingua-language-detector>=2.2.0"

# miservice：用 PyPI 3.0.1 + 本地补丁修复 OTP 登录（ConfirmPhone 页面 70016 问题）。
# 补丁来源：Yonsm/MiService PR #64（2026-10-15 已合并）。PyPI 3.0.1 尚未含此修复。
# slim 镜像不带 patch 命令，需先装（用 Debian 镜像源）
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir "miservice==3.0.1"
COPY patches/miservice-otp-fix.patch /tmp/miservice-otp-fix.patch
RUN cd /usr/local/lib/python3.12/site-packages \
    && patch -p1 < /tmp/miservice-otp-fix.patch \
    && rm /tmp/miservice-otp-fix.patch

# 本项目源码（含 Agent Loop / 动态日期 / 反幻觉提示词 / 请稍等补丁）
COPY xiaogpt /app/xiaogpt
COPY xiaogpt.py /app/xiaogpt.py
COPY xiaogpt/agent_bot.py /app/agent_bot.py

VOLUME /config
EXPOSE 9527
ENTRYPOINT ["python", "-m", "xiaogpt.cli"]
