# ZAAM Project Deployment Guide

## Option 1: Deploying to Expo's Hosting Service (EAS)

1. Install EAS CLI:
   ```
   npm install -g eas-cli
   ```

2. Login to Expo:
   ```
   npx eas login
   ```

3. Configure the project for EAS:
   ```
   npx eas build:configure
   ```

4. Build the web project:
   ```
   npx eas build --platform web
   ```

5. Deploy the web project:
   ```
   npx eas deploy --platform web
   ```

## Option 2: Manual Deployment to Static Hosting

Since the Expo webpack configuration is having issues with the current project setup, you can use the following approach:

1. Start the Expo development server in web mode:
   ```
   expo start --web
   ```

2. Once the development server is running, open a browser and load the web application.

3. Use browser tools (Right-click -> "Save As" or "Save Page As") to save the complete webpage and its resources.

4. Alternatively, use browser extensions like "SingleFile" that can save an entire web page as a single HTML file with all resources embedded.

5. Upload the saved files to any static hosting service like:
   - Netlify
   - Vercel
   - GitHub Pages
   - Firebase Hosting
   - AWS S3

## Option 3: Use Specialized Expo Hosting

For Expo projects specifically:

1. Create an account on [Expo.dev](https://expo.dev/)
2. Run:
   ```
   npx expo publish
   ```

This will make your app available on the Expo hosting service.

## Known Issues

The current project has some configuration challenges that prevent a standard webpack build. This might be due to:

1. Dependencies on specific React Native components that don't have web equivalents
2. Configuration specificities in the Expo project
3. Webpack compatibility issues with some libraries

If you need a proper production build, consider:
1. Consulting with an Expo expert
2. Updating the project dependencies
3. Migrating to a more web-friendly configuration 