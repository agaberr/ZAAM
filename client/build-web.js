const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('Building web application...');

try {
  // Create web-build directory if it doesn't exist
  const webBuildDir = path.join(__dirname, 'web-build');
  if (!fs.existsSync(webBuildDir)) {
    fs.mkdirSync(webBuildDir, { recursive: true });
  }

  // Build the web app
  console.log('Starting Expo web build...');
  
  // We'll modify process.env to include what's needed for the web build
  process.env.NODE_ENV = 'production';
  
  execSync('npx expo start --web --no-dev --minify --https=false', {
    stdio: 'inherit',
    env: {
      ...process.env,
      EXPO_WEB_BUILD_DIR: webBuildDir
    }
  });
  
  console.log('Build complete! Files are in the ./web-build directory');
} catch (error) {
  console.error('Build failed:', error);
  process.exit(1);
} 