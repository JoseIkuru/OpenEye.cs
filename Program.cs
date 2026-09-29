using System;
using OpenCvSharp;

namespace OpenEye
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load the pre-trained tracking models
            // Ensure these XML files are in your main project folder!
            using var faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
            using var eyeCascade = new CascadeClassifier("haarcascade_eye.xml");

            if (faceCascade.Empty() || eyeCascade.Empty())
            {
                Console.WriteLine("Error: Could not load cascade XML files. Check paths!");
                return;
            }

            // 2. Open up your webcam (0 is usually the default built-in camera)
            using var capture = new VideoCapture(0);
            if (!capture.IsOpened())
            {
                Console.WriteLine("Error: Could not open webcam video stream.");
                return;
            }

            // Allocate temporary canvas spaces for video processing
            using var frame = new Mat();
            using var grayFrame = new Mat();

            // Create a designated rendering window
            using var window = new Window("OpenEye Tracker (Press ESC to exit)");

            Console.WriteLine("Eye tracker active! Click on the video window and hit ESC to close.");

            // 3. Main Real-Time Processing Loop
            while (true)
            {
                capture.Read(frame); // Grab the current video frame
                if (frame.Empty()) break;

                // OpenCV processes pattern recognition significantly faster in grayscale
                Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);
                Cv2.EqualizeHist(grayFrame, grayFrame); // Boosts contrast for clearer lighting

                // 4. Scan for Faces First (isolates our search area to save computing power)
                Rect[] faces = faceCascade.DetectMultiScale(grayFrame, scaleFactor: 1.1, minNeighbors: 5, minSize: new Size(30, 30));

                foreach (var faceRect in faces)
                {
                    // Draw a subtle blue box around the detected face
                    Cv2.Rectangle(frame, faceRect, Scalar.FromRgb(0, 0, 255), thickness: 2);

                    // Crop the grayscale image down to JUST the face zone to scan for eyes
                    using Mat faceZoneGray = new Mat(grayFrame, faceRect);
                    using Mat faceZoneColor = new Mat(frame, faceRect);

                    // 5. Scan for Eyes strictly inside the face zone
                    Rect[] eyes = eyeCascade.DetectMultiScale(faceZoneGray, scaleFactor: 1.1, minNeighbors: 10, minSize: new Size(15, 15));

                    foreach (var eyeRect in eyes)
                    {
                        // Draw a bright green box around each eye
                        Cv2.Rectangle(faceZoneColor, eyeRect, Scalar.FromRgb(0, 255, 0), thickness: 2);
                    }
                }

                // Show the modified color frame inside our window
                window.ShowImage(frame);

                // Wait 30 milliseconds; if user hits the ESC key (ASCII value 27), break out
                if (Cv2.WaitKey(30) == 27)
                {
                    break;
                }
            }
        }
    }
}
