# Lightweight production container for VectorDB
FROM node:20-alpine

WORKDIR /app

COPY package.json ./
COPY server.js ./
COPY index.html ./
COPY README.md ./

EXPOSE 8080

CMD ["node", "server.js"]
