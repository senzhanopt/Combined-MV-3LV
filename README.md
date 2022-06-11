***Combined MV-LV Power Grid Operation: Comparing Sequential, Integrated, and Decentralized Control Architectures***
Author: S. Zhan, J. Morren, W. v. d. Akker, A. v. d. Molen, N. Paterakis, J. G. Slootweg
Electrical Energy Systems, Department of Electrical Engineering, Eindhoven University of Technology
Contact Information: s.zhan@tue.nl

***Data Collection***
The dataset includes topology data and profiles of base load and PVs. The source is the Simbench project [1].

[1] S. Meinecke, D. Sarajli´c, S. R. Drauz, A. Klettke, L.-P. Lauven, C. Rehtanz, A. Moser, and M. Braun, “Simbench—a benchmark dataset
of electric power systems to compare innovative solutions based on power flow analysis,” Energies, vol. 13, no. 12, p. 3290, Jun. 2020.

***Code Specifications***
*.xlsx: data as the file name suggests;
opf_mv_lv: integrated;
opf_separated: sequential;
opf_admm, opf_benders, opf_app: decentralized;
pf_mv_lv: power flow program;
timeSeries: perform time series simulation using above functions.