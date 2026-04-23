package com.eatwise.mobile

import android.app.AlertDialog
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.core.content.FileProvider
import android.webkit.CookieManager
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File

class MainActivity : ComponentActivity() {

    private lateinit var webView: WebView
    private var filePathCallback: ValueCallback<Array<Uri>>? = null
    private var pendingCameraImageUri: Uri? = null

    private val filePickerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val callback = filePathCallback
        if (callback == null) {
            return@registerForActivityResult
        }

        val dataUri: Uri? = result.data?.data
        val finalUri = dataUri ?: pendingCameraImageUri
        if (finalUri != null) {
            callback.onReceiveValue(arrayOf(finalUri))
        } else {
            callback.onReceiveValue(null)
        }
        pendingCameraImageUri = null
        filePathCallback = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        webView = WebView(this)
        setContentView(webView)

        val prefs = getSharedPreferences("eatwise_prefs", MODE_PRIVATE)
        val savedUrl = prefs.getString("app_url", "")?.trim().orEmpty()

        if (savedUrl.isEmpty()) {
            promptForUrl("https://your-streamlit-app-url")
        }

        configureWebView()
        val appUrl = prefs.getString("app_url", "https://your-streamlit-app-url")?.trim().orEmpty()
        webView.loadUrl(appUrl)
    }

    private fun configureWebView() {
        val settings: WebSettings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.allowFileAccess = true
        settings.mediaPlaybackRequiresUserGesture = false
        settings.setSupportZoom(false)
        settings.cacheMode = WebSettings.LOAD_DEFAULT

        CookieManager.getInstance().setAcceptCookie(true)
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, true)

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                return false
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest?) {
                request?.grant(request.resources)
            }

            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                this@MainActivity.filePathCallback?.onReceiveValue(null)
                this@MainActivity.filePathCallback = filePathCallback

                val contentSelectionIntent = Intent(Intent.ACTION_GET_CONTENT)
                contentSelectionIntent.addCategory(Intent.CATEGORY_OPENABLE)
                contentSelectionIntent.type = "image/*"

                val cameraImageFile = createTempImageFile()
                val cameraImageUri = FileProvider.getUriForFile(
                    this@MainActivity,
                    "${packageName}.fileprovider",
                    cameraImageFile
                )
                pendingCameraImageUri = cameraImageUri

                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, cameraImageUri)
                cameraIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                cameraIntent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)

                val chooserIntent = Intent(Intent.ACTION_CHOOSER)
                chooserIntent.putExtra(Intent.EXTRA_INTENT, contentSelectionIntent)
                chooserIntent.putExtra(Intent.EXTRA_TITLE, "Select label image")
                chooserIntent.putExtra(Intent.EXTRA_INITIAL_INTENTS, arrayOf(cameraIntent))

                return try {
                    filePickerLauncher.launch(chooserIntent)
                    true
                } catch (e: ActivityNotFoundException) {
                    this@MainActivity.filePathCallback = null
                    false
                }
            }
        }
    }

    private fun createTempImageFile(): File {
        return File.createTempFile("eatwise_capture_", ".jpg", cacheDir)
    }

    private fun promptForUrl(defaultValue: String) {
        val input = android.widget.EditText(this)
        input.setText(defaultValue)

        AlertDialog.Builder(this)
            .setTitle("Set App URL")
            .setMessage("Enter your deployed Streamlit URL")
            .setView(input)
            .setCancelable(false)
            .setPositiveButton("Save") { _, _ ->
                val url = input.text.toString().trim().ifEmpty { defaultValue }
                getSharedPreferences("eatwise_prefs", MODE_PRIVATE)
                    .edit()
                    .putString("app_url", url)
                    .apply()
            }
            .show()
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }
}
