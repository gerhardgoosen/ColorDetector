package application;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import opencv.ColorBlobDetector;
import utils.Utils;

public class ColorDetectorController {

	// FXML camera button
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// the FXML area for showing the mask
	@FXML
	private ImageView maskImage;
	// the FXML area for showing the output of the morphological operations
	@FXML
	private ImageView morphImage;
	// FXML slider for setting HSV ranges
	@FXML
	private Slider redRange;
	@FXML
	private Slider greenRange;
	@FXML
	private Slider blueRange;
	@FXML
	private Slider alphaRange;

	// FXML label to show the current values set with the sliders
	@FXML
	private Label rgbCurrentValues;

	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture = new VideoCapture();
	// a flag to change the button behavior
	private boolean cameraActive;

	private boolean mIsColorSelected = false;
	// private Mat mRgba;
	private ColorBlobDetector mDetector;
	private Scalar selectedColor;
	private Mat mSpectrum;
	private Size SPECTRUM_SIZE;
	private Scalar CONTOUR_COLOR;

	// property for object binding
	private ObjectProperty<String> rgbValuesProp;

	// ColorBlobDetectionActivity colorDetectActivity;

	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	private void startCamera() {

		// bind a text property with the string containing the current range of
		// HSV values for object detection
		rgbValuesProp = new SimpleObjectProperty<>();
		this.rgbCurrentValues.textProperty().bind(rgbValuesProp);

		// set a fixed width for all the image to show and preserve image ratio
		this.imageViewProperties(this.originalFrame, 600);

		mDetector = new ColorBlobDetector();
		mSpectrum = new Mat();
		SPECTRUM_SIZE = new Size(200, 64);
		CONTOUR_COLOR = new Scalar(255, 0, 0, 255);

		if (!this.cameraActive) {
			// start the video capture
			this.capture.open(0);

			// is the video stream available?
			if (this.capture.isOpened()) {
				this.cameraActive = true;

				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {

					@Override
					public void run() {
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(originalFrame, imageToShow);
					}
				};

				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

				// update the button content
				this.cameraButton.setText("Stop Camera");
			} else {
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		} else {
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Start Camera");

			// stop the timer
			this.stopAcquisition();

		}
	}

	@FXML
	private void sliderActioned() {
		// show the current selected HSV range
		String valuesToPrint = "Red: " + redRange.getValue() + "\tGreen: " + greenRange.getValue() + "\tBlue: "
				+ blueRange.getValue() + "\t Aplha: " + alphaRange.getValue();

		if(rgbValuesProp==null){
			rgbValuesProp = new SimpleObjectProperty<>();
			this.rgbCurrentValues.textProperty().bind(rgbValuesProp);
		}
		Utils.onFXThread(this.rgbValuesProp, valuesToPrint);
	}

	/**
	 * Get a frame from the opened video stream (if any)
	 *
	 * @return the {@link Image} to show
	 */
	private Mat grabFrame() {
		Mat frame = new Mat();

		// check if the capture is open
		if (this.capture.isOpened()) {
			try {
				// read the current frame
				this.capture.read(frame);

				// if the frame is not empty, process it
				if (!frame.empty()) {

					this.setColorToDetect(new Scalar(blueRange.getValue(), greenRange.getValue(), redRange.getValue(),
							alphaRange.getValue()));

					if (mIsColorSelected) {
						mDetector.process(frame);
						List<MatOfPoint> contours = mDetector.getContours();
						// System.out.println("Contours count: " +
						// contours.size());
						Imgproc.drawContours(frame, contours, -1, CONTOUR_COLOR);

						Mat colorLabel = frame.submat(4, 68, 4, 68);
						colorLabel.setTo(getSelectedColor());

						Mat spectrumLabel = frame.submat(4, 4 + mSpectrum.rows(), 70, 70 + mSpectrum.cols());
						mSpectrum.copyTo(spectrumLabel);
					}

				}

			} catch (Exception e) {
				// log the (full) error
				System.err.print("Exception during the image elaboration...");
				e.printStackTrace();
			}
		}

		return frame;
	}

	/**
	 * Set typical {@link ImageView} properties: a fixed width and the
	 * information to preserve the original image ration
	 *
	 * @param image
	 *            the {@link ImageView} to use
	 * @param dimension
	 *            the width of the image to set
	 */
	private void imageViewProperties(ImageView image, int dimension) {
		// set a fixed width for the given ImageView
		image.setFitWidth(dimension);
		// preserve the image ratio
		image.setPreserveRatio(true);
	}

	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition() {
		if (this.timer != null && !this.timer.isShutdown()) {
			try {
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			} catch (InterruptedException e) {
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}

		if (this.capture.isOpened()) {
			// release the camera
			this.capture.release();

		}
	}

	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 *
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image) {
		Utils.onFXThread(view.imageProperty(), image);
	}

	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed() {
		this.stopAcquisition();
	}

	public void setColorToDetect(Scalar selectedColor) {
		setSelectedColor(selectedColor);

		// System.out.println("Selected rgba color: (" + selectedColor.val[0] +
		// ", " + selectedColor.val[1] + ", "
		// + selectedColor.val[2] + ", " + selectedColor.val[3] + ")");

		mDetector.setHsvColor(selectedColor);

		Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);

		mIsColorSelected = true;

	}

	public Scalar converScalarHsv2Rgba(Scalar hsvColor) {
		Mat pointMatRgba = new Mat();
		Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
		Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

		return new Scalar(pointMatRgba.get(0, 0));
	}

	public Scalar getSelectedColor() {
		return selectedColor;
	}

	public void setSelectedColor(Scalar selectedColor) {
		this.selectedColor = selectedColor;
	}
}