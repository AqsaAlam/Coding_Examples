"""

The purpose of this file is to convert yeast systematic gene names to 
yeast systematic gene names with the full descriptions.

Example input:
    
YBL014C
YAL001C
YBL015W
YAL002W
YBL016W

Example output:
    
YBL014C ; RRN6 ; Component of the core factor (CF) rDNA transcription factor complex
YAL001C ; TFC3 ; Subunit of RNA polymerase III transcription initiation factor complex
YBL015W ; ACH1 ; Protein with CoA transferase activity
YAL002W ; VPS8 ; Membrane-binding component of the CORVET complex
YBL016W ; FUS3 ; Mitogen-activated serine/threonine protein kinase involved in mating

"""

# Open the various files
file1 = open("full_yeast_descriptions.txt", "r") # yeast descriptions
file2 = open("fix_me.txt", "r") # to change (input)
output = open("fixed.txt", "w") # changed (output)


# Make a dictionary of the descriptions
# So each systematic gene name is associated with a long description
descriptions = {}
for line in file1:
    gene = (line.rstrip()).split(";")[0].strip()
    descriptions[gene] = line.rstrip()

# Replace the systematic gene name with the long description
# Write it to the new file
for line in file2:
    if line.rstrip()[::] in descriptions:
        replace = descriptions[line.rstrip()[::]]
        output.write(replace)
        output.write("\n")
    else:
        output.write(line.rstrip()[::])
        output.write("\n")

# Close the various files 
file1.close()
file2.close()
output.close()


