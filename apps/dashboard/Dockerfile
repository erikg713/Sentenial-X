FROM node:18-alpine
WORKDIR /app
COPY apps/dashboard/package.json .
RUN npm install
COPY apps/dashboard/ .
EXPOSE 3000
CMD ["npm", "start"]
