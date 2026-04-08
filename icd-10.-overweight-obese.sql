-- ICD-10 CODES FOR OBESITY/overweight
-- SOURCE: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1047.501/expansion/Latest
-- https://www.icd10data.com/ICD10CM/Codes/Z00-Z99/Z68-Z68/Z68-
-- https://pmc.ncbi.nlm.nih.gov/articles/PMC8578343/#:~:text=Obesity%20classification%20from%20claims%20data,9%20and%20ICD%2D10%20codes.

DECLARE condition_obese_overweight ARRAY<STRING>;

SET condition_obese_overweight = ['E6601', 'E6609', 'E661', 'E662', 'E663', 'E66811', 'E66812', 'E66813', 'E6689', 'E669', 'E660', 'E6609', 'Z6825', 'Z6826', 'Z6827', 'Z6828', 'Z6829', 'Z6830', 'Z6831', 'Z6832', 'Z6833', 'Z6834', 'Z6835', 'Z6836', 'Z6837', 'Z6838', 'Z6839', 'Z684', 'Z6841', 'Z6842', 'Z6843', 'Z6844', 'Z6845'];