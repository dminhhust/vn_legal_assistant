# React frontend (replaces the old Streamlit frontend). Multi-stage:
# node builds the static bundle, nginx serves it and proxies /api to
# the FastAPI backend. Works both in docker-compose (backend on the
# `backend` service) and standalone (BACKEND_PROXY defaults to
# localhost:8080).
FROM node:20-slim AS build

WORKDIR /build

COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . .
RUN npm run build

FROM nginx:alpine

COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /build/dist /usr/share/nginx/html

# The official nginx image envsubst-processes /etc/nginx/templates at
# startup; these are the only two knobs the template needs.
ENV FRONTEND_PORT=8501 \
    BACKEND_PROXY=http://localhost:8080

EXPOSE 8501