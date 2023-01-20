
# This is to generate a FASTA file for every gene to have multiple sequences.

# ==========================================================================

import requests, sys, time
from time import sleep
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# ==========================================================================

# FILE OF HOMOLOGOUS SPECIES NAMES

fungi = []
file1 = "0_all_species_names_edited.txt"
for line in open(file1):
    fungi.append(line.rstrip())

# FILE OF YEAST GENES TO INCLUDE

genes = []
file2 = "4337_gene_systematic_names.txt"
for line in open(file2):
    genes.append(line.rstrip())


# ==========================================================================

# FOR RETRIEVING ORTHOLOGS OF A GENE (GIVEN SYSTEMATIC GENE NAME)


retry_strategy1 = Retry(
total=3,
status_forcelist=[429, 500, 502, 503, 504],
method_whitelist=["HEAD", "GET", "OPTIONS"]
)
    
adapter = HTTPAdapter(max_retries=retry_strategy1)
s = requests.Session()

def homolog_list(gene, s):
    """
    imports required: requests, sys
    gene: str, s: session object
    """
    
    server = "https://rest.ensembl.org"
    ext = "/homology/id/" + gene + "?compara=fungi;type=orthologues"
     
    
    r = s.get(server+ext, headers={ "Content-Type" : "application/json"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()
     
    decoded = r.json()

    species = []
    IDs = []

    for organism in decoded["data"][0]["homologies"]:
        if organism["type"] == "ortholog_one2one":
            if organism["target"]["species"] in fungi:
                species.append(organism["target"]["species"])
                IDs.append(organism["target"]["id"])

# lists are respective (ex. index 5 in species corresponds to index 5 in IDs)
    
    return (species, IDs)

# ==========================================================================

# FOR RETRIEVING SEQUENCE OF A GENE (GIVEN SYSTEMATIC GENE NAME)


retry_strategy2 = Retry(
total=3,
status_forcelist=[429, 500, 502, 503, 504],
method_whitelist=["HEAD", "GET", "OPTIONS"]
)

adapter = HTTPAdapter(max_retries=retry_strategy2)
t = requests.Session()

def yeast_sequence(yeast_gene):

     
    server = "https://rest.ensembl.org"
    ext = r"/sequence/id/" + yeast_gene + "?type=genomic;expand_3prime=200"
     
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
     
    if not r.ok:
      r.raise_for_status()
      sys.exit()
     
    sequence = r.text
    return sequence[len(sequence)-200::]

    
def sequence(gene, ids, species, t): # WILL ALSO MAKE THE FILE!
    """
    gene is the YEAST gene
    ids is a list of the ids of the OTHER SPECIES
    species is a list of the names of the OTHER SPECIES 
    t is the session
    """

    server = "http://rest.ensembl.org"
    ext = "/sequence/id?type=genomic;expand_3prime=200"
    headers={ "Content-Type" : "text/plain", "Accept" : "application/json"}

    dct = {"ids": ids}
    dct = json.dumps(dct)
    r = t.post(server+ext, headers=headers, data=dct)


    if not r.ok:
        r.raise_for_status()
        sys.exit()
        
    file = open(gene + ".txt", "w")
    
    for i in range(len(r.json())):# for every sequence in the list in the order of the ids 
        file.write(">" + ids[i] + " " + species[i] + "\n" + r.json()[i]["seq"][len(r.json()[i]["seq"])-200::] + "\n")

    #Response.close()
    file.close()
# ==========================================================================

y = 0
for gene in genes:
    y += 1
    if y == 50:
        time.sleep(1)
        y = 0
    homologs = homolog_list(gene, t)
    species = homologs[0]
    ids = homologs[1]
    if species:
        sequence(gene, ids, species, s)
        file = open(gene + ".txt", "a")
        file.write(">" + gene + " " + "saccharomyces_cerevisiae" + "\n" + yeast_sequence(gene))
        file.close()
















