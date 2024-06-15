# Constrained Multiobjective Optimisation (CMO)

This project contains Python implementations of several CMO benchmark suites, including the benchmark suite of collated problems, called STEP. The details of the suites and the problems they contain are provided at the bottom of this readme.

The project also contains implementations of the Indicator for Constrained Multi-Objective Problems (ICMOP), the Empirical Runtime Distribution Function (ECDF) for ICMOP, and a Pymoo callback for tracking the runtime.

The problems are designed to work in much the same manner as they do in Pymoo (https://pymoo.org/index.html). Therefore, they work in the Pymoo `minimize` function, with Pymoo algorithms.

## Getting Started

Install this package via pip using `pip install cmo`.

## Dependencies

### Problems Package

- For MODAct suite: https://github.com/epfl-lamd/modact

### Indicator Package

- Pygmo: see installation guide here: https://esa.github.io/pygmo2/install.html

## Example Use

For accessing a single problem:
```python
from cmo.problems.factory import get_problem

problem_name = 'c1dtlz1' # The alias is used (see below)
n_obj, n_var = 3, 10

problem = get_problem(problem_name, n_obj=n_obj, n_var=n_var)
```

For using the STEP suite:
```python
from cmo.problems.factory import get_problem

for i in range(1, 31):

    problem_name = f'step{i}'

    problem = get_problem(problem_name)
```

For accessing a large set of continuous CMOPs:
```python
from cmo.problems.factory import get_problem, get_problem_list

for problem_name in get_problem_list():
    
    problem = get_problem(problem_name)
```

For integrating with Pymoo and using the indicators:
```python
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from cmo.indicator.ecdf import ECDF
from cmo.indicator.icmop import ICMOP
from cmo.indicator.runtime_callback import RuntimeCallback
from cmo.problems.factory import get_problem

problem = get_problem('step1')

icmop = ICMOP(problem)

res = minimize(problem,
               algorithm=NSGA2(),
               termination=('n_evals', 10000),
               verbose=False,
               callback=RuntimeCallback(icmop))

ecdf = ECDF(runtimes=[res.algorithm.callback.runtime],
            tau_ref=0,  # This value is used to offset the targets
            eps=np.linspace(2, -1, 100))

print('Area under the ECDF curve:', ecdf.compute_auc())

ecdf.visualise()
```

For accessing all problems noted in the list below, under CMOPs:
```python
from cmo.problems.factory import get_problem, get_all_problem_list

for problem_name in get_all_problem_list():

    problem = get_problem(problem_name)
```

## Citation

If you use this repository in your work, please cite it using the following:

J. N. Cork, T. Tušar, and B. Filipič, “Test problem selection from benchmark suites in constrained multiobjective optimisation,” Arxiv, 2024.

```
@article{Cork2024,
    author = {Jordan N. Cork and Tea Tu\v{s}ar and Bogdan Filipi\v{c}},
    title = {Test Problem Selection from Benchmark Suites in Constrained Multiobjective Optimisation},
    journal = {Arxiv},
    year = {2024}
}
```

## CMOPs

| Suite          | CMOP           | Alias         | Variables | Objectives | Reference |
|:---------------|:---------------|:--------------|:---------:|:----------:|:---------:|
| C-DTLZ         | C1-DTLZ1       | c1dtlz1       |   ≥2,≥M   |     ≥2     |    [1]    |
| C-DTLZ         | C1-DTLZ3       | c1dtlz3       |   ≥2,≥M   |     ≥2     |    [1]    |
| C-DTLZ         | C2-DTLZ2       | c2dtlz2       |   ≥2,≥M   |     ≥2     |    [1]    |
| C-DTLZ         | ConvexC2-DTLZ2 | convexc2dtlz2 |   ≥2,≥M   |     ≥2     |    [1]    |
| C-DTLZ         | C3-DTLZ1       | c3dtlz1       |   ≥2,≥M   |     ≥2     |    [1]    |
| C-DTLZ         | C3-DTLZ4       | c3dtlz4       |   ≥3,≥M   |     ≥2     |    [1]    |
| CF             | CF1            | cf1           |    ≥3     |     2      |    [2]    |
| CF             | CF2            | cf2           |    ≥3     |     2      |    [2]    |
| CF             | CF3            | cf3           |    ≥3     |     2      |    [2]    |
| CF             | CF4            | cf4           |    ≥3     |     2      |    [2]    |
| CF             | CF5            | cf5           |    ≥3     |     2      |    [2]    |
| CF             | CF6            | cf6           |    ≥4     |     2      |    [2]    |
| CF             | CF7            | cf7           |    ≥4     |     2      |    [2]    |
| CF             | CF8            | cf8           |    ≥5     |     3      |    [2]    |
| CF             | CF9            | cf9           |    ≥5     |     3      |    [2]    |
| CF             | CF10           | cf10          |    ≥5     |     3      |    [2]    |
| CTP            | CTP1           | ctp1          |    ≥2     |     2      |    [3]    |
| CTP            | CTP2           | ctp2          |    ≥2     |     2      |    [3]    |
| CTP            | CTP3           | ctp3          |    ≥2     |     2      |    [3]    |
| CTP            | CTP4           | ctp4          |    ≥2     |     2      |    [3]    |
| CTP            | CTP5           | ctp5          |    ≥2     |     2      |    [3]    |
| CTP            | CTP6           | ctp6          |    ≥2     |     2      |    [3]    |
| CTP            | CTP7           | ctp7          |    ≥2     |     2      |    [3]    |
| CTP            | CTP8           | ctp8          |    ≥2     |     2      |    [3]    |
| CRE            | CRE21          | cre21         |     3     |     2      |    [9]    |
| CRE            | CRE22          | cre22         |     4     |     2      |    [9]    |
| CRE            | CRE23          | cre23         |     4     |     2      |    [9]    |
| CRE            | CRE24          | cre24         |     7     |     2      |    [9]    |
| CRE            | CRE25          | cre25         |     4     |     2      |    [9]    |
| CRE            | CRE31          | cre31         |     7     |     3      |    [9]    |
| CRE            | CRE32          | cre32         |     6     |     3      |    [9]    |
| CRE            | CRE51          | cre51         |     3     |     5      |    [9]    |
| DAS-CMOP       | DAS-CMOP1      | dascmop1      |    ≥2     |     2      |    [4]    |
| DAS-CMOP       | DAS-CMOP2      | dascmop2      |    ≥2     |     2      |    [4]    |
| DAS-CMOP       | DAS-CMOP3      | dascmop3      |    ≥2     |     2      |    [4]    |
| DAS-CMOP       | DAS-CMOP4      | dascmop4      |    ≥2     |     2      |    [4]    |
| DAS-CMOP       | DAS-CMOP5      | dascmop5      |    ≥2     |     2      |    [4]    |
| DAS-CMOP       | DAS-CMOP6      | dascmop6      |    ≥2     |     2      |    [4]    |
| DAS-CMOP       | DAS-CMOP7      | dascmop7      |    ≥3     |     3      |    [4]    |
| DAS-CMOP       | DAS-CMOP8      | dascmop8      |    ≥3     |     3      |    [4]    |
| DAS-CMOP       | DAS-CMOP9      | dascmop9      |    ≥3     |     3      |    [4]    |
| DC-DTLZ        | DC1-DTLZ1      | dc1dtlz1      |   ≥2,≥M   |     ≥2     |    [5]    |
| DC-DTLZ        | DC1-DTLZ3      | dc1dtlz3      |   ≥2,≥M   |     ≥2     |    [5]    |
| DC-DTLZ        | DC2-DTLZ1      | dc2dtlz1      |   ≥2,≥M   |     ≥2     |    [5]    |
| DC-DTLZ        | DC2-DTLZ3      | dc2dtlz3      |   ≥2,≥M   |     ≥2     |    [5]    |
| DC-DTLZ        | DC3-DTLZ1      | dc3dtlz1      |   ≥2,≥M   |     ≥2     |    [5]    |
| DC-DTLZ        | DC3-DTLZ3      | dc3dtlz3      |   ≥2,≥M   |     ≥2     |    [5]    |
| DOC            | DOC1           | doc1          |     6     |     2      |    [18]   |
| DOC            | DOC2           | doc2          |    16     |     2      |    [18]   |
| DOC            | DOC3           | doc3          |    10     |     2      |    [18]   |
| DOC            | DOC4           | doc4          |     8     |     2      |    [18]   |
| DOC            | DOC5           | doc5          |     8     |     2      |    [18]   |
| DOC            | DOC6           | doc6          |    11     |     2      |    [18]   |
| DOC            | DOC7           | doc7          |    11     |     2      |    [18]   |
| DOC            | DOC8           | doc8          |    10     |     3      |    [18]   |
| DOC            | DOC9           | doc9          |    11     |     3      |    [18]   |
| LIR-CMOP       | LIR-CMOP1      | lircmop1      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP2      | lircmop2      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP3      | lircmop3      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP4      | lircmop4      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP5      | lircmop5      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP6      | lircmop6      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP7      | lircmop7      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP8      | lircmop8      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP9      | lircmop9      |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP10     | lircmop10     |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP11     | lircmop11     |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP12     | lircmop12     |    30     |     2      |    [6]    |
| LIR-CMOP       | LIR-CMOP13     | lircmop13     |    30     |     3      |    [6]    |
| LIR-CMOP       | LIR-CMOP14     | lircmop14     |    30     |     3      |    [6]    |
| MODAct         | CS1            | cs1           |    20     |     2      |    [16]   |
| MODAct         | CS2            | cs2           |    20     |     2      |    [16]   |
| MODAct         | CS3            | cs3           |    20     |     2      |    [16]   |
| MODAct         | CS4            | cs4           |    20     |     2      |    [16]   |
| MODAct         | CT1            | ct1           |    20     |     2      |    [16]   |
| MODAct         | CT2            | ct2           |    20     |     2      |    [16]   |
| MODAct         | CT3            | ct3           |    20     |     2      |    [16]   |
| MODAct         | CT4            | ct4           |    20     |     2      |    [16]   |
| MODAct         | CTS1           | cts1          |    20     |     3      |    [16]   |
| MODAct         | CTS2           | cts2          |    20     |     3      |    [16]   |
| MODAct         | CTS3           | cts3          |    20     |     3      |    [16]   |
| MODAct         | CTS4           | cts4          |    20     |     3      |    [16]   |
| MODAct         | CTSE1          | ctse1         |    20     |     4      |    [16]   |
| MODAct         | CTSE2          | ctse2         |    20     |     4      |    [16]   |
| MODAct         | CTSE3          | ctse3         |    20     |     4      |    [16]   |
| MODAct         | CTSE4          | ctse4         |    20     |     4      |    [16]   |
| MODAct         | CTSEI1         | ctsei1        |    20     |     5      |    [16]   |
| MODAct         | CTSEI2         | ctsei2        |    20     |     5      |    [16]   |
| MODAct         | CTSEI3         | ctsei3        |    20     |     5      |    [16]   |
| MODAct         | CTSEI4         | ctsei4        |    20     |     5      |    [16]   |
| MW             | MW1            | mw1           |    ≥3     |     2      |    [7]    |
| MW             | MW2            | mw2           |    ≥3     |     2      |    [7]    |
| MW             | MW3            | mw3           |    ≥2     |     2      |    [7]    |
| MW             | MW4            | mw4           |    ≥3     |     ≥2     |    [7]    |
| MW             | MW5            | mw5           |    ≥3     |     2      |    [7]    |
| MW             | MW6            | mw6           |    ≥2     |     2      |    [7]    |
| MW             | MW7            | mw7           |    ≥2     |     2      |    [7]    |
| MW             | MW8            | mw8           |    ≥2     |     ≥2     |    [7]    |
| MW             | MW9            | mw9           |    ≥3     |     2      |    [7]    |
| MW             | MW10           | mw10          |    ≥2     |     2      |    [7]    |
| MW             | MW11           | mw11          |    ≥2     |     2      |    [7]    |
| MW             | MW12           | mw12          |    ≥3     |     2      |    [7]    |
| MW             | MW13           | mw13          |    ≥2     |     2      |    [7]    |
| MW             | MW14           | mw14          |    ≥2     |     ≥2     |    [7]    |
| NCTP           | NCTP1          | nctp1         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP2          | nctp2         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP3          | nctp3         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP4          | nctp4         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP5          | nctp5         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP6          | nctp6         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP7          | nctp7         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP8          | nctp8         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP9          | nctp9         |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP10         | nctp10        |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP11         | nctp11        |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP12         | nctp12        |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP13         | nctp13        |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP14         | nctp14        |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP15         | nctp15        |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP16         | nctp16        |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP17         | nctp17        |    ≥3     |     2      |    [8]    |
| NCTP           | NCTP18         | nctp18        |    ≥3     |     2      |    [8]    |
| RCM            | RCM1           | rcm1          |     4     |     2      |    [10]   |
| RCM            | RCM2           | rcm2          |     5     |     2      |    [10]   |
| RCM            | RCM3           | rcm3          |     3     |     2      |    [10]   |
| RCM            | RCM4           | rcm4          |     4     |     2      |    [10]   |
| RCM            | RCM5           | rcm5          |     4     |     2      |    [10]   |
| RCM            | RCM6           | rcm6          |     7     |     2      |    [10]   |
| RCM            | RCM7           | rcm7          |     4     |     2      |    [10]   |
| RCM            | RCM8           | rcm8          |     7     |     3      |    [10]   |
| RCM            | RCM10          | rcm10         |     2     |     2      |    [10]   |
| RCM            | RCM11          | rcm11         |     3     |     5      |    [10]   |
| RCM            | RCM12          | rcm12         |     4     |     2      |    [10]   |
| RCM            | RCM13          | rcm13         |     7     |     3      |    [10]   |
| RCM            | RCM14          | rcm14         |     5     |     2      |    [10]   |
| RCM            | RCM15          | rcm15         |     3     |     2      |    [10]   |
| RCM            | RCM16          | rcm16         |     2     |     2      |    [10]   |
| RCM            | RCM17          | rcm17         |     6     |     3      |    [10]   |
| RCM            | RCM18          | rcm18         |     3     |     2      |    [10]   |
| RCM            | RCM19          | rcm19         |    10     |     3      |    [10]   |
| RCM            | RCM20          | rcm20         |     4     |     2      |    [10]   |
| RCM            | RCM21          | rcm21         |     6     |     2      |    [10]   |
| RCM            | RCM25          | rcm25         |     2     |     2      |    [10]   |
| RCM            | RCM27          | rcm27         |     3     |     2      |    [10]   |
| RCM            | RCM29          | rcm29         |     7     |     2      |    [10]   |
| Classic        | BNH            | bnh           |     2     |     2      |    [11]   |
| Classic        | TNK            | tnk           |     2     |     2      |    [12]   |
| Classic        | SRN            | srn           |     2     |     2      |    [13]   |
| Classic        | OSY            | osy           |     6     |     2      |    [14]   |
| Classic        | WB             | wb            |     4     |     2      |    [15]   |
| ZXHCF          | ZXHCF1         | zxhcf1        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF2         | zxhcf2        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF3         | zxhcf3        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF4         | zxhcf4        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF5         | zxhcf5        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF6         | zxhcf6        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF7         | zxhcf7        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF8         | zxhcf8        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF9         | zxhcf9        |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF10        | zxhcf10       |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF11        | zxhcf11       |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF12        | zxhcf12       |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF13        | zxhcf13       |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF14        | zxhcf14       |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF15        | zxhcf15       |   ≥2,>M   |    ≥2      |    [17]   |
| ZXHCF          | ZXHCF16        | zxhcf16       |   ≥2,>M   |    ≥2      |    [17]   |

## References

[1] H. Jain, K. Deb, An evolutionary many-objective optimization algorithm using reference-point 
based nondominated sorting approach, part II: Handling constraints and extending to an adaptive 
approach, IEEE Trans. Evol. Comput. 18 (4) (2014) 602–622. doi:10.1109/TEVC.2013.2281534.

[2] Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, S. Tiwari, Multiobjective optimization 
test instances for the CEC 2009 special session and competition, Technical report CES-487, The 
School of Computer Science and Electronic Engieering, University of Essex, Colchester, UK (2008).

[3] K. Deb, A. Pratap, T. Meyarivan, Constrained test problems for multiobjective evolutionary 
optimization, in: International Conference on Evolutionary Multi-Criterion Optimization (EMO 2001), 
2001, pp. 284–298.

[4] Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, E. Goodman, Difficulty adjustable and 
scalable constrained multiobjective test problem toolkit, Evol. Comput. 28 (3) (2019) 339–378. 
doi:10.1162/evco-a-00259.

[5] K. Li, R. Chen, G. Fu, X. Yao, Two-archive evolutionary algorithm for constrained multiobjective 
optimization, IEEE Trans. Evol. Comput. 23 (2) (2019) 303–315. doi:10.1109/TEVC.2018.2855411.

[6] Z. Fan, W. Li, X. Cai, H. Huang, Y. Fang, Y. You, J. Mo, C. Wei, E. Goodman, An improved epsilon 
constraint-handling method in MOEA/D for CMOPs with large infeasible regions, Soft Comput. 
23 (23) (2019) 12491–12510. doi:10.1007/s00500-019-03794-x.

[7] Z. Ma, Y. Wang, Evolutionary constrained multiobjective optimization: Test suite construction 
and performance comparisons, IEEE Trans. Evol. Comput. 23 (6) (2019) 972–986. 
doi:10.1109/TEVC.2019.2896967.

[8] J. P. Li, Y. Wang, S. Yang, Z. Cai, A comparative study of constraint-handling techniques in 
evolutionary constrained multiobjective optimization, in: 2016 IEEE Congress on Evolutionary 
Computation (CEC), 2016, pp. 4175–4182. doi:10.1109/CEC.2016.7744320.

[9] R. Tanabe, H. Ishibuchi, An Easy-to-use Real-world Multi-objective Problem Suite, Applied 
Soft Computing. 89: 106078 (2020). doi:10.1016/j.asoc.2020.106078.

[10] A. Kumar, G. Wu, M. Z. Ali, Q. Luo, R. Mallipeddi, P. N. Suganthan, S. Das, A Benchmark-Suite 
of real-World constrained multi-objective optimization problems and some baseline results, Swarm 
and Evolutionary Computation, Volume 67, 2021, 100961. doi:10.1016/j.swevo.2021.100961.

[11] T. T. Binh and U. Korn, Mobes: a multiobjective evolution strategy for constrained optimization problems, 
in: Proceedings of the Third International Conference on Genetic Algorithms (MENDEL97, 176–182. 1997.

[12] J. Blank, TNK, 2022, https://pymoo.org/problems/multi/tnk.html, Accessed 27 September 2023.

[13] J. Blank, SRN, 2022, https://github.com/anyoptimization/pymoo/blob/main/pymoo/problems/multi/srn.py, 
Accessed 27 September 2023.

[14] J. Blank, OSY, 2022, https://pymoo.org/problems/multi/osy.html, Accessed 27 September 2023.

[15] J. Blank, Welded Beam, 2022, https://pymoo.org/problems/multi/welded_beam.html, Accessed 27 September 2023.

[16] C. Picard and J. Schiffmann, “Realistic Constrained Multi-Objective Optimization Benchmark Problems
from Design,” IEEE Transactions on Evolutionary Computation, pp. 1–1, 2020.

[17] Y. Zhou, Y. Xiang, and X. He, Constrained multiobjective optimization: Test problem construction and 
performance evaluations, IEEE Transactions on Evolutionary Computation, 2021, 25(1): 172-186.

[18] Z. Liu and Y. Wang, Handling constrained multiobjective optimization problems with constraints 
in both the decision and objective spaces. IEEE Transactions on Evolutionary Computation, 2019, 23(5): 870-884.
