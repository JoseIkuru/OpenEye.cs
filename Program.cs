using System;
using OpenCvSharp;

namespace OpenEye
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load the pre-trained tracking models
            using var faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
            using var eyeCascade = new CascadeClassifier("haarcascade_eye.xml");

            if (faceCascade.Empty() || eyeCascade.Empty())
            {
                Console.WriteLine("Error: Could not load cascade XML files.");
                return;
            }

            // 2. Open up your webcam
            using var capture = new VideoCapture(0);
            if (!capture.IsOpened())
            {
                Console.WriteLine("Error: Could not open webcam video stream.");
                return;
            }

            // Read a baseline frame to gather camera width/height specs
            using var frame = new Mat();
            capture.Read(frame);
            if (frame.Empty()) return;

            // 3. Setup the Heat Map Accumulator Canvas
            // We initialize a blank black image matching your webcam's resolution (Single Channel, 8-bit)
            Mat heatAccumulator = new Mat(frame.Size(), MatType.CV_8UC1, Scalar.All(0));

            // Create separate rendering windows
            using var videoWindow = new Window("Real-Time Tracking Feed");
            using var heatWindow = new Window("Live Eye Focus Heat Map");

            using var grayFrame = new Mat();

            Console.WriteLine("Tracking active! Focus your eyes in the screen. Press ESC to exit.");

            // 4. Main Real-Time Processing Loop
            while (true)
            {
                capture.Read(frame);
                if (frame.Empty()) break;

                Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);
                Cv2.EqualizeHist(grayFrame, grayFrame);

                // Scan for Faces
                // REPLACED Size.Empty WITH new Size()
                Rect[] faces = faceCascade.DetectMultiScale(grayFrame, scaleFactor: 1.1, minNeighbors: 5, flags: HaarDetectionTypes.ScaleImage, minSize: new Size(30, 30));

                foreach (var faceRect in faces)
                {
                    Cv2.Rectangle(frame, faceRect, Scalar.FromRgb(0, 0, 255), 2);

                    using Mat faceZoneGray = new Mat(grayFrame, faceRect);
                    using Mat faceZoneColor = new Mat(frame, faceRect);

                    // Scan for Eyes inside the face area
                    // REPLACED the parameters to match the mandatory signature order
                    Rect[] eyes = eyeCascade.DetectMultiScale(faceZoneGray, scaleFactor: 1.1, minNeighbors: 10, flags: HaarDetectionTypes.ScaleImage, minSize: new Size(15, 15));

                    foreach (var eyeRect in eyes)
                    {
                        // 1. Crop down to JUST the detected eye box area
                        using Mat eyeZoneGray = new Mat(faceZoneGray, eyeRect);
                        using Mat eyeZoneColor = new Mat(faceZoneColor, eyeRect);

                        // 2. Erase the top 25% of the eye box (Removes dark eyelashes/eyebrows that confuse the tracker)
                        int upperCrop = eyeRect.Height / 4;
                        Rect cropArea = new Rect(0, upperCrop, eyeRect.Width, eyeRect.Height - upperCrop);
                        using Mat croppedEyeGray = new Mat(eyeZoneGray, cropArea);
                        using Mat croppedEyeColor = new Mat(eyeZoneColor, cropArea);

                        // 3. Apply Thresholding (Turns everything white EXCEPT the darkest values like the pupil)
                        using Mat thresholdMat = new Mat();
                        // Adjust '30' higher if your room is dark, lower if your room is very bright
                        Cv2.Threshold(croppedEyeGray, thresholdMat, thresh: 30, maxval: 255, ThresholdTypes.BinaryInv);

                        // 4. Find the contours (shapes) of the dark spots left behind
                        Cv2.FindContours(thresholdMat, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                        Point pupilCenter = new Point(-1, -1);

                        // 5. Look for the most circular shape (your pupil)
                        foreach (var contour in contours)
                        {
                            var moments = Cv2.Moments(contour);
                            if (moments.M00 > 10) // Filter out tiny speckles/noise
                            {
                                // Calculate the exact center mass of the dark spot
                                int pupilX = (int)(moments.M10 / moments.M00);
                                int pupilY = (int)(moments.M01 / moments.M00);
                                
                                pupilCenter = new Point(pupilX, pupilY);
                                
                                // Draw a tiny neon dot directly over the pupil in the video feed
                                Cv2.Circle(croppedEyeColor, pupilCenter, radius: 3, color: Scalar.FromRgb(255, 0, 0), thickness: -1);
                                break; // Found the primary dark mass, break out
                            }
                        }

                        // 6. Project the pupil's precise moving coordinates back to the global Heat Map canvas
                        if (pupilCenter.X != -1 && pupilCenter.Y != -1)
                        {
                            int globalPupilX = faceRect.X + eyeRect.X + pupilCenter.X;
                            int globalPupilY = faceRect.Y + eyeRect.Y + upperCrop + pupilCenter.Y;

                            // Draw onto the heat map canvas with a much tighter radius for precision
                            Cv2.Circle(heatAccumulator, new Point(globalPupilX, globalPupilY), radius: 8, color: Scalar.All(4), thickness: -1);
                        }
                    }
                }

                // 7. Post-Process the Accumulator Canvas into a Thermal Glow
                using Mat blurredHeat = new Mat();
                using Mat colorHeatMap = new Mat();
                using Mat outputOverlay = new Mat();

                if (!heatAccumulator.Empty())
                {
                    // Smooth the absolute dots out using a Gaussian Blur to simulate a smooth distribution dropoff
                    Cv2.GaussianBlur(heatAccumulator, blurredHeat, new Size(45, 45), 0);

                    // Map the pixel intensity scale from Grayscale (0-255) to a Thermal color layout (Jet = Blue to Red)
                    Cv2.ApplyColorMap(blurredHeat, colorHeatMap, ColormapTypes.Jet);

                    // Blend the heat map smoothly over your webcam frame at a 60% transparency level
                    Cv2.AddWeighted(frame, 0.6, colorHeatMap, 0.4, 0, outputOverlay);
                    
                    // Display the resulting visualizations
                    heatWindow.ShowImage(colorHeatMap);
                    videoWindow.ShowImage(outputOverlay);
                }

                // Wait 30 milliseconds; exit if ESC key (ASCII 27) is hit
                if (Cv2.WaitKey(30) == 27)
                {
                    break;
                }
            }

            // Cleanup explicitly tracked unmanaged Mat matrices on closure
            heatAccumulator.Dispose();
        }
    }
}
