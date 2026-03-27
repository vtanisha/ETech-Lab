# Stage 1: Build the React Application
FROM node:20-alpine AS build

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
# Pass required environment variables during build phase
ARG VITE_TABLEAU_REPORT_URL
ENV VITE_TABLEAU_REPORT_URL=$VITE_TABLEAU_REPORT_URL

RUN npm run build

# Stage 2: Serve the Built Static Files via Nginx
FROM nginx:alpine

# Copy custom Nginx configuration to handle React Router client-side history API routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy Vite build output directly into Nginx web root directory
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
