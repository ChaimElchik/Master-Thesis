def clean_file(input_file):
    output_file = input_file.replace('.txt', '_clean.txt')

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write header to the output file
        outfile.write("frame,id,x,y,x_offset,y_offset\n")

        # Process each line of the input file
        for line in infile:
            # Split the line by commas
            parts = line.strip().split(',')

            # Extract relevant information and convert to desired format
            frame = parts[0]
            ID = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            x_offset = float(parts[4])
            y_offset = float(parts[5])

            # Write the cleaned data to the output file
            outfile.write(f"{frame},{ID},{x},{y},{x_offset},{y_offset}\n")

    print(f"Cleaned data saved to {output_file}")
