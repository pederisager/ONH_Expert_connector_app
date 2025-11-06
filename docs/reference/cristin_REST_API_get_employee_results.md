# Link to Cristin REST API documentation
https://api.cristin.no/v2/doc/index.html

# Procedure for pulling all NVA results data for a single individual

1. for each employee, identify their NVA person ID. It is the number at the end of their NVA profile link in staff.csv. For example, Peder Mortvedt Isager has NVA link https://nva.sikt.no/research-profile/618557, and so his person ID is 618557. 

2. Find the the JSON file containing all NVA results listed for an employee by going to https://api.cristin.no/v2/persons/<person_ID>/results. For example, all results for Peder Mortvedt Isager are listed under https://api.cristin.no/v2/persons/618557/results. 

3. Store all information from the JSON file that will be relevant for an employee. This includes information such as result title, result summary, journal title for academic articles, and so on. All information not relevant to this app should be removed/not stored in order to preserve the quality of the RAG query matching process later on. 