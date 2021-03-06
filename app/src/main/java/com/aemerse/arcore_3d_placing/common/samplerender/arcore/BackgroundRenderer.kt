package com.aemerse.arcore_3d_placing.common.samplerender.arcore

import android.media.Image
import android.opengl.GLES30
import com.google.ar.core.*
import com.aemerse.arcore_3d_placing.common.samplerender.*
import com.aemerse.arcore_3d_placing.common.samplerender.Shader.BlendFactor
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * This class both renders the AR camera background and composes the a scene foreground. The camera
 * background can be rendered as either camera image data or camera depth data. The virtual scene
 * can be composited with or without depth occlusion.
 */
class BackgroundRenderer(render: SampleRender?) {
    companion object {
        private val TAG = BackgroundRenderer::class.java.simpleName

        // components_per_vertex * number_of_vertices * float_size
        private const val COORDS_BUFFER_SIZE = 2 * 4 * 4
        private val NDC_QUAD_COORDS_BUFFER = ByteBuffer.allocateDirect(COORDS_BUFFER_SIZE).order(
            ByteOrder.nativeOrder()).asFloatBuffer()
        private val VIRTUAL_SCENE_TEX_COORDS_BUFFER =
            ByteBuffer.allocateDirect(COORDS_BUFFER_SIZE).order(
                ByteOrder.nativeOrder()).asFloatBuffer()

        init {
            NDC_QUAD_COORDS_BUFFER.put(floatArrayOf( /*0:*/
                -1f, -1f,  /*1:*/+1f, -1f,  /*2:*/-1f, +1f,  /*3:*/+1f, +1f))
            VIRTUAL_SCENE_TEX_COORDS_BUFFER.put(floatArrayOf( /*0:*/
                0f, 0f,  /*1:*/1f, 0f,  /*2:*/0f, 1f,  /*3:*/1f, 1f))
        }
    }

    private val cameraTexCoords =
        ByteBuffer.allocateDirect(COORDS_BUFFER_SIZE).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val mesh: Mesh
    private val cameraTexCoordsVertexBuffer: VertexBuffer
    private var backgroundShader: Shader? = null
    private var occlusionShader: Shader? = null
    /** Return the camera depth texture generated by this object.  */
    val cameraDepthTexture: Texture
    /** Return the camera color texture generated by this object.  */
    val cameraColorTexture: Texture
    private var depthColorPaletteTexture: Texture? = null
    private var useDepthVisualization = false
    private var useOcclusion = false
    private var aspectRatio = 0f

    /**
     * Sets whether the background camera image should be replaced with a depth visualization instead.
     * This reloads the corresponding shader code, and must be called on the GL thread.
     */
    @Throws(IOException::class)
    fun setUseDepthVisualization(render: SampleRender, useDepthVisualization: Boolean) {
        if (backgroundShader != null) {
            if (this.useDepthVisualization == useDepthVisualization) {
                return
            }
            backgroundShader!!.close()
            backgroundShader = null
            this.useDepthVisualization = useDepthVisualization
        }
        if (useDepthVisualization) {
            depthColorPaletteTexture = Texture.createFromAsset(
                render,
                "models/depth_color_palette.png",
                Texture.WrapMode.CLAMP_TO_EDGE,
                Texture.ColorFormat.LINEAR)
            backgroundShader = Shader.createFromAssets(
                render,
                "shaders/background_show_depth_color_visualization.vert",
                "shaders/background_show_depth_color_visualization.frag",  /*defines=*/
                null)
                .setTexture("u_CameraDepthTexture", cameraDepthTexture)
                .setTexture("u_ColorMap", depthColorPaletteTexture!!)
                .setDepthTest(false)
                .setDepthWrite(false)
        } else {
            backgroundShader = Shader.createFromAssets(
                render,
                "shaders/background_show_camera.vert",
                "shaders/background_show_camera.frag",  /*defines=*/
                null)
                .setTexture("u_CameraColorTexture", cameraColorTexture)
                .setDepthTest(false)
                .setDepthWrite(false)
        }
    }

    /**
     * Sets whether to use depth for occlusion. This reloads the shader code with new `#define`s, and must be called on the GL thread.
     */
    @Throws(IOException::class)
    fun setUseOcclusion(render: SampleRender, useOcclusion: Boolean) {
        if (occlusionShader != null) {
            if (this.useOcclusion == useOcclusion) {
                return
            }
            occlusionShader!!.close()
            occlusionShader = null
            this.useOcclusion = useOcclusion
        }
        val defines = HashMap<String, String>()
        defines["USE_OCCLUSION"] = if (useOcclusion) "1" else "0"
        occlusionShader = Shader.createFromAssets(render,
            "shaders/occlusion.vert",
            "shaders/occlusion.frag",
            defines)
            .setDepthTest(false)
            .setDepthWrite(false)
            .setBlend(BlendFactor.SRC_ALPHA, BlendFactor.ONE_MINUS_SRC_ALPHA)
        if (useOcclusion) {
            occlusionShader!!.setTexture("u_CameraDepthTexture", cameraDepthTexture)
                .setFloat("u_DepthAspectRatio", aspectRatio)
        }
    }

    /**
     * Updates the display geometry. This must be called every frame before calling either of
     * BackgroundRenderer's draw methods.
     *
     * @param frame The current `Frame` as returned by [Session.update].
     */
    fun updateDisplayGeometry(frame: Frame) {
        if (frame.hasDisplayGeometryChanged()) {
            // If display rotation changed (also includes view size change), we need to re-query the UV
            // coordinates for the screen rect, as they may have changed as well.
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                NDC_QUAD_COORDS_BUFFER,
                Coordinates2d.TEXTURE_NORMALIZED,
                cameraTexCoords)
            cameraTexCoordsVertexBuffer.set(cameraTexCoords)
        }
    }

    /** Update depth texture with Image contents.  */
    fun updateCameraDepthTexture(image: Image) {
        // SampleRender abstraction leaks here
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, cameraDepthTexture.textureId[0])
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_RG8,
            image.width,
            image.height,
            0,
            GLES30.GL_RG,
            GLES30.GL_UNSIGNED_BYTE,
            image.planes[0].buffer)
        if (useOcclusion) {
            aspectRatio = image.width.toFloat() / image.height.toFloat()
            occlusionShader!!.setFloat("u_DepthAspectRatio", aspectRatio)
        }
    }

    /**
     * Draws the AR background image. The image will be drawn such that virtual content rendered with
     * the matrices provided by [com.google.ar.core.Camera.getViewMatrix] and
     * [com.google.ar.core.Camera.getProjectionMatrix] will
     * accurately follow static physical objects.
     */
    fun drawBackground(render: SampleRender) {
        render.draw(mesh, backgroundShader!!)
    }

    /**
     * Draws the virtual scene. Any objects rendered in the given [Framebuffer] will be drawn
     * given the previously specified [OcclusionMode].
     *
     *
     * Virtual content should be rendered using the matrices provided by [ ][com.google.ar.core.Camera.getViewMatrix] and [ ][com.google.ar.core.Camera.getProjectionMatrix].
     */
    fun drawVirtualScene(
        render: SampleRender, virtualSceneFramebuffer: Framebuffer, zNear: Float, zFar: Float
    ) {
        occlusionShader!!.setTexture(
            "u_VirtualSceneColorTexture", virtualSceneFramebuffer.colorTexture!!)
        if (useOcclusion) {
            occlusionShader!!.setTexture("u_VirtualSceneDepthTexture", virtualSceneFramebuffer.depthTexture!!)
                .setFloat("u_ZNear", zNear)
                .setFloat("u_ZFar", zFar)
        }
        render.draw(mesh, occlusionShader!!)
    }

    /**
     * Allocates and initializes OpenGL resources needed by the background renderer. Must be called
     * during a [SampleRender.Renderer] callback, typically in [ ][SampleRender.Renderer.onSurfaceCreated].
     */
    init {
        cameraColorTexture = Texture(
            render,
            Texture.Target.TEXTURE_EXTERNAL_OES,
            Texture.WrapMode.CLAMP_TO_EDGE,  /*useMipmaps=*/
            false)
        cameraDepthTexture = Texture(
            render,
            Texture.Target.TEXTURE_2D,
            Texture.WrapMode.CLAMP_TO_EDGE,  /*useMipmaps=*/
            false)

        // Create a Mesh with three vertex buffers: one for the screen coordinates (normalized device
        // coordinates), one for the camera texture coordinates (to be populated with proper data later
        // before drawing), and one for the virtual scene texture coordinates (unit texture quad)
        val screenCoordsVertexBuffer =
            VertexBuffer(render,  /* numberOfEntriesPerVertex=*/2, NDC_QUAD_COORDS_BUFFER)
        cameraTexCoordsVertexBuffer =
            VertexBuffer(render,  /*numberOfEntriesPerVertex=*/2,  /*entries=*/null)
        val virtualSceneTexCoordsVertexBuffer =
            VertexBuffer(render,  /* numberOfEntriesPerVertex=*/2, VIRTUAL_SCENE_TEX_COORDS_BUFFER)
        val vertexBuffers = arrayOf(
            screenCoordsVertexBuffer,
            cameraTexCoordsVertexBuffer,
            virtualSceneTexCoordsVertexBuffer)
        mesh = Mesh(render, Mesh.PrimitiveMode.TRIANGLE_STRIP,  /*indexBuffer=*/null, vertexBuffers)
    }
}