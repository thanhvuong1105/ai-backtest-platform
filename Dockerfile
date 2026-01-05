# ===== Base image =====
FROM python:3.10-slim

# ===== Install NodeJS =====
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean

# ===== Working directory =====
WORKDIR /app

# ===== Copy package.json & install node deps =====
COPY package*.json ./
RUN npm install

# ===== Copy entire project =====
COPY . .

# ===== Install python deps =====
RUN pip install --no-cache-dir pandas numpy

# ===== Expose API port =====
EXPOSE 3001

# ===== Start server =====
CMD ["node", "api/server.js"]
