import cv2
import pandas as pd

def TR_Vid_Creator(input_video,output_video,df):
    # Open the input video file
    cap = cv2.VideoCapture(input_video)

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Iterate through each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of the video, break the loop
            break

        # Get the bounding box coordinates for the current frame
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_boxes = df[df['frame'] == frame_number]

        # Draw bounding boxes on the frame
        for _, row in frame_boxes.iterrows():
            x, y, x_offset, y_offset, obj_id = int(row['x']), int(row['y']), int(row['x-offset']), int(row['y-offset']), int(row['id'])
            xmin, ymin, xmax, ymax = x, y, x + x_offset, y + y_offset

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Display object id
            cv2.putText(frame, f"ID: {obj_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    # Release the video capture and VideoWriter, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
