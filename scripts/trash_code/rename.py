
train_names = '../../dataset/train.txt'
val_names='../../dataset/validation.txt'
test_names='../../dataset/test.txt'

files=(train_names, val_names, test_names)

# Open the input and output files
for input_file in files:
    with open(input_file, 'r') as input_f, open(input_file+'2', 'w') as output_f:
        # Read each line from the input file
        for line in input_f:
            # Extract the filename from the line
            filename = line.strip().split('/')[-1]
            
            # Write the modified line to the output file
            output_f.write(filename + '\n')

print('File conversion complete.')