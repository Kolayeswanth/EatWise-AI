# EatWise Mobile Android Wrapper

This Android app wraps your existing Streamlit assistant so mobile behavior stays exactly the same as your web app.

## What It Does

- Opens your deployed Streamlit app inside a mobile WebView.
- Supports image upload and camera chooser from file inputs.
- Saves the app URL on first launch so you can point to your Streamlit deployment.

## Folder

- Android project root: mobile-android
- Main activity: app/src/main/java/com/eatwise/mobile/MainActivity.kt

## Build APK (Android Studio)

1. Install Android Studio (JDK included).
2. Open the folder mobile-android as a project.
3. Let Gradle sync and install requested SDK components.
4. Build APK:
   - Build -> Build Bundle(s) / APK(s) -> Build APK(s)
5. APK output path:
   - app/build/outputs/apk/debug/app-debug.apk

## Build APK (CLI)

After Android SDK and Gradle are installed and available:

1. Open terminal in mobile-android
2. Run:
   - gradle assembleDebug
3. APK output:
   - app/build/outputs/apk/debug/app-debug.apk

## First Launch

- App asks for your Streamlit URL.
- Enter your deployed frontend URL and save.
- The app will remember it for next launches.

## Notes

- This wrapper keeps your backend APIs and ML logic unchanged.
- For production release signing, configure signing in app/build.gradle.
