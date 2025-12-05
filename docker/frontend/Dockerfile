FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY frontend/ ./

# Expose port
EXPOSE 5173

# Start development server
CMD ["npm", "run", "dev"]
