-- ICD-10 CODES FOR underweight or normal weight
-- SOURCE: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1108.99/expansion/Latest
-- https://www.icd10data.com/ICD10CM/Codes/Z00-Z99/Z68-Z68/Z68-
-- https://pmc.ncbi.nlm.nih.gov/articles/PMC8578343/#:~:text=Obesity%20classification%20from%20claims%20data,9%20and%20ICD%2D10%20codes.

DECLARE condition_underweight_normal ARRAY<STRING>;

SET condition_underweight_normal = ['R636, 'Z681', 'Z6820', 'Z6821', 'Z6822', 'Z6823', 'Z6824'];