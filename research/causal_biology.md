# Causal biology and task-aware enhancement: source dossier

Researched 2026-10-05. Scope: public sources supporting Week 26; event context,
effect identification, multi-environment discovery, invariant prediction, measurement,
and task-aware enhancement. This is not an evaluation of Synlico's proprietary system.

## Event and provenance

The [official SF Tech Week event listing](https://www.tech-week.com/calendar/sf/events/causal-ai-for-drug-discovery-learning-from-patient-data-sf-techweek-5ccd23ec-afbe-4206-99a0-3208dd443476)
lists Synlico's *Causal AI for Drug Discovery: Learning from Patient Data* for October 5,
2026, 1–2 PM, virtual, with the topic of structure across environments and task-aware
enhancement. The user's supplied description identifies moderator Jingwei Lu and speakers
Yingzhen Yang and Amin Jaber. We did not access a talk recording or unpublished slides.
Do not present the lesson as reporting what the speakers said.

[Synlico's technology page](https://synlico.com/technology/) describes patient-context
causal modeling and a proprietary enhancement model. These are company descriptions;
they do not supply an independently audited performance benchmark or a reproducible
enhancement algorithm. Dataset-scale claims can change, so the lesson avoids treating
them as a scientific result.

[Yingzhen Yang's ASU profile](https://search.asu.edu/profile/3520440) and
[author homepage](https://yingzhenyang.com/) establish statistical learning, optimization,
and sparse/subspace research. The public sources reviewed do not identify an exact method
behind the event's phrase “task-aware data enhancement.” The lesson therefore uses a
clearly labeled nuisance-randomization teaching heuristic and a separate public task-driven
augmentation paper, without attributing either to Yang or Synlico.

## 1. Effect identification before estimation

Miguel A. Hernán and James M. Robins (2020), *Causal Inference: What If*.
Chapman & Hall/CRC. [Author-hosted book and resources](https://miguelhernan.org/whatifbook).
Consult chapters 1–3 (effects and observational assumptions), 6–9 (graphs, confounding,
selection and measurement), and 13 (standardization). The author page supplies a living
online version; the 2020 citation is the authors' suggested reference.

Lesson use: define population, treatment versions, outcomes, timing, consistency,
exchangeability and positivity before calculating an ATE. Our binary-stratum standardization
is an elementary teaching estimator. It omits survival outcomes, censoring, longitudinal
confounding, and target-trial emulation. The law-of-total-expectation derivation in Week 26
is supplied by the course; this citation does not prove assumptions for any dataset.

## 2. Multiple domains and latent confounding

Adam Li, Amin Jaber, Elias Bareinboim (2023), *Causal discovery from observational and
interventional data across multiple environments*. Advances in Neural Information
Processing Systems 36, pp. 16942–16956. DOI 10.52202/075280-0741.
[Proceedings abstract](https://proceedings.neurips.cc/paper_files/paper/2023/hash/368cba57d00902c752eaa9e4770bbbbe-Abstract-Conference.html),
[paper PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/368cba57d00902c752eaa9e4770bbbbe-Paper-Conference.pdf).
No arXiv ID was verified; the official proceedings are the version cited.

The paper develops cross-domain causal constraints using selection diagrams, S-Markov
properties, and S-FCI, including latent-confounded systems. Its equivalence between
multi-domain observational learning and unknown-target interventions is a formal result
within a stated model; it is not a blanket claim that biological batches are randomized
experiments. Week 26 explains the distinction but does not implement S-FCI or reproduce
its experiments. Partial orientations and unidentified effects remain possible.

## 3. Invariant causal prediction

Jonas Peters, Peter Bühlmann, Nicolai Meinshausen (2016), *Causal inference by using
invariant prediction: identification and confidence intervals*. Journal of the Royal
Statistical Society, Series B 78(5), 947–1012. DOI 10.1111/rssb.12167.
arXiv:1501.01332. [Author preprint](https://arxiv.org/abs/1501.01332),
[publication PDF](https://people.math.ethz.ch/~peterbu/Files/Manuscripts/invariant-causal-prediction.pdf).

ICP concerns predictors of a target whose conditional mechanism is invariant across the
specified environments under assumptions. Intersecting suitably accepted sets can provide
error-controlled parent statements with valid tests; the environment richness affects
identifiability. Our residual mean/variance spread is a descriptive comparison, not an ICP
implementation, confidence set, or hypothesis test. Stable prediction alone does not certify
a full causal graph, and target interventions can invalidate invariance.

For broader context, Joris M. Mooij, Sara Magliacane, Tom Claassen (2020), *Joint Causal
Inference from Multiple Contexts*, JMLR 21(99), 1–108, arXiv:1611.10351.
[Official article](https://jmlr.org/papers/v21/17-123.html).
JCI is a framework that uses context variables and background assumptions, not one fixed
algorithm. Its discussion supplies context for PC, FCI, equivalence-class output, and
the distinction between a context label and a controlled intervention.

## 4. A public task-driven enhancement example

Krishna Chaitanya, Neerav Karani, Christian Baumgartner, Olivio Donati,
Anton Becker, Ender Konukoglu (2019), *Semi-Supervised and Task-Driven Data Augmentation*.
IPMI 2019, LNCS 11492, pp. 29–41. DOI 10.1007/978-3-030-20351-1_3.
arXiv:1902.05396. [Preprint](https://arxiv.org/abs/1902.05396),
[ETH institutional publication record](https://www.research-collection.ethz.ch/handle/20.500.11850/377309?show=full).
The author ordering above follows the cited arXiv version.

This work optimizes generated transformations for a segmentation task and keeps image and
mask transformations aligned. The transferable lesson is to specify downstream utility and
label preservation when enhancing data. Medical-image augmentation evidence does not
establish gene-regulatory graph recovery or efficacy in patient-derived omics. Our miniature
randomization heuristic does not reproduce its networks, losses, or experiments.

## 5. Measurement and false signals after imputation

Tallulah S. Andrews, Martin Hemberg (2018; revised version 2 published 2019), *False signals
induced by single-cell imputation*. F1000Research 7:1740.
DOI 10.12688/f1000research.16613.2. No arXiv ID verified.
[Versioned primary article](https://f1000research.com/articles/7-1740),
[PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6415334/).

The study uses negative controls to assess false signals that imputation can introduce.
For our lesson, this motivates assessing correlation/edge artifacts separately from a
reconstruction or prediction objective. It is not a theorem that every imputation always
hurts every task. The explicit two-marker smoother counterexample in Week 26 is constructed
by the course; no particular published method is claimed to be that smoother.

## Additional speaker context and validation route

Yingzhen Yang and Ping Li (2022), *Noisy L0-sparse subspace clustering on dimensionality
reduced data*. UAI 2022, PMLR 180:2235–2245, arXiv:2206.11079.
[Official proceedings](https://proceedings.mlr.press/v180/yang22e.html).
The paper concerns noisy subspace detection and dimensionality reduction under stated
models. It is useful context about noise and structural assumptions; it is not cited as
causal identification or as the undisclosed event enhancement algorithm.

Joseph M. Replogle, Reuben A. Saunders, Angela N. Pogson, Jeffrey A. Hussmann,
Alexander Lenail, Alina Guna, Lauren Mascibroda, Eric J. Wagner, Karen Adelman,
Gila Lithwick-Yanai, Nika Iremadze, Florian Oberstrass, Doron Lipson, Jessica L. Bonnar,
Marco Jost, Thomas M. Norman, Jonathan S. Weissman (2022), *Mapping information-rich
genotype-phenotype landscapes with genome-scale Perturb-seq*. Cell 185(14), 2559–2575.e28.
DOI 10.1016/j.cell.2022.05.013.
[Primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/).
This experimental perturbation study illustrates a route for testing molecular hypotheses;
its cell contexts and readouts do not establish patient treatment effects. Perturbation
specificity, timing, controls, and cell states still define what an experiment can validate.

## Course boundaries and overlap audit

Weeks 11/12/14/25 already develop support, OPE, exposure, and audit reasoning.
Week 26 adds graph/SCM language, observational equivalence, cross-environment mechanisms,
patient grouping, measurement artifacts, and task-specific augmentation. ATE standardization
is presented as a short bridge, not another duplicate IPS assignment.

The project uses synthetic continuous measurements with explicit MCAR missingness and
known nuisance identity. It implements NumPy regression, a training-only transformation,
patient splits and bootstrap, and effect standardization. It implements none of PC, FCI,
S-FCI, ICP, scRNA-seq count modeling, biological target discovery, or clinical decision support.
Reported numbers are simulator outcomes with seed/configuration, not clinical evidence.
The source dossier distinguishes researched findings from instructor-created examples.
