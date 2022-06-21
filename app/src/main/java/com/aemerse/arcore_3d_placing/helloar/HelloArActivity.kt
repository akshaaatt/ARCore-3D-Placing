package com.aemerse.arcore_3d_placing.helloar

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Config
import com.google.ar.core.Config.InstantPlacementMode
import com.google.ar.core.Session
import com.aemerse.arcore_3d_placing.common.helpers.CameraPermissionHelper
import com.aemerse.arcore_3d_placing.common.helpers.DepthSettings
import com.aemerse.arcore_3d_placing.common.helpers.FullScreenHelper
import com.aemerse.arcore_3d_placing.common.helpers.InstantPlacementSettings
import com.aemerse.arcore_3d_placing.common.samplerender.SampleRender
import com.aemerse.arcore_3d_placing.common.helpers.ARCoreSessionLifecycleHelper
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException

/**
 * This is a simple example that shows how to create an augmented reality (AR) application using the
 * ARCore API. The application will display any detected planes and will allow the user to tap on a
 * plane to place a 3D model.
 */
class HelloArActivity : AppCompatActivity() {
  companion object {
    private const val TAG = "HelloArActivity"
  }

  lateinit var arCoreSessionHelper: ARCoreSessionLifecycleHelper
  lateinit var view: HelloArView
  lateinit var renderer: HelloArRenderer

  val instantPlacementSettings = InstantPlacementSettings()
  val depthSettings = DepthSettings()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // Setup ARCore session lifecycle helper and configuration.
    arCoreSessionHelper = ARCoreSessionLifecycleHelper(this)
    // If Session creation or Session.resume() fails, display a message and log detailed
    // information.
    arCoreSessionHelper.exceptionCallback =
      { exception ->
        val message =
          when (exception) {
            is UnavailableUserDeclinedInstallationException ->
              "Please install Google Play Services for AR"
            is UnavailableApkTooOldException -> "Please update ARCore"
            is UnavailableSdkTooOldException -> "Please update this app"
            is UnavailableDeviceNotCompatibleException -> "This device does not support AR"
            is CameraNotAvailableException -> "Camera not available. Try restarting the app."
            else -> "Failed to create AR session: $exception"
          }
        Log.e(TAG, "ARCore threw an exception", exception)
        view.snackbarHelper.showError(this, message)
      }

    // Configure session features, including: Lighting Estimation, Depth mode, Instant Placement.
    arCoreSessionHelper.beforeSessionResume = ::configureSession
    lifecycle.addObserver(arCoreSessionHelper)

    // Set up the Hello AR renderer.
    renderer = HelloArRenderer(this)
    lifecycle.addObserver(renderer)

    // Set up Hello AR UI.
    view = HelloArView(this)
    lifecycle.addObserver(view)
    setContentView(view.root)

    // Sets up an example renderer using our HelloARRenderer.
    SampleRender(view.surfaceView, renderer, assets)

    depthSettings.onCreate(this)
    instantPlacementSettings.onCreate(this)
  }

  // Configure the session, using Lighting Estimation, and Depth mode.
  fun configureSession(session: Session) {
    session.configure(
      session.config.apply {
        lightEstimationMode = Config.LightEstimationMode.ENVIRONMENTAL_HDR

        // Depth API is used if it is configured in Hello AR's settings.
        depthMode = when {
              session.isDepthModeSupported(Config.DepthMode.AUTOMATIC) -> {
                  Config.DepthMode.AUTOMATIC
              }
              else -> {
                  Config.DepthMode.DISABLED
              }
          }

        // Instant Placement is used if it is configured in Hello AR's settings.
        instantPlacementMode =
          when {
              instantPlacementSettings.isInstantPlacementEnabled() -> {
                  InstantPlacementMode.LOCAL_Y_UP
              }
              else -> {
                  InstantPlacementMode.DISABLED
              }
          }
      }
    )
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, results: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, results)
    if (!CameraPermissionHelper.hasCameraPermission(this)) {
      // Use toast instead of snackbar here since the activity will exit.
      Toast.makeText(this, "Camera permission is needed to run this application", Toast.LENGTH_LONG)
        .show()
      if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
        // Permission denied with checking "Do not ask again".
        CameraPermissionHelper.launchPermissionSettings(this)
      }
      finish()
    }
  }

  override fun onWindowFocusChanged(hasFocus: Boolean) {
    super.onWindowFocusChanged(hasFocus)
    FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus)
  }
}
