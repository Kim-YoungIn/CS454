# Genetic Algorithms for Adversarial Examples

## 1. Approach and Evaluation

I implemented genetic algorithms to generate adversarial examples for a pretrained VGG-11 classifier. The objective is to change the model's original prediction while minimizing the difference from the input image. The same classifier and evaluation procedure are used for all images, including inputs that were not provided with the assignment. I used model predictions without gradients, and I did not use saved attacks or image-specific labels in the search functions.

For an original image x, I take the model's top prediction as the reference class y. Candidate images are evaluated using the log-probability margin

$$m(x') = \log p_y(x') - \max_{k \ne y}\log p_k(x').$$

A negative margin means that a different class has become the most likely prediction. I retain the smallest successful perturbation separately from the current population, so later mutations cannot discard an already valid result. The search uses selection, crossover, and mutation, but the representation and refinement procedure differ across the three metrics.

I use the assignment's actual distance functions. In particular, its $L_2$ score is the **sum of squared tensor differences**, not their square root, and its $L_0$ score counts **changed channel elements**, not RGB pixels. Norm attacks use a 165-second internal CPU budget, below the three-minute limit. The optional fog search uses 270 seconds, below its five-minute limit.

## 2. GA Implementation

**$L_\infty$.** I represent perturbations using signed spatial blocks. This allows the search to alter many tensor elements while controlling their largest absolute change. The GA keeps four selected candidates, combines their perturbations, and introduces new spatial mutations. After finding a successful attack, it reduces the allowed radius and continues searching. The final refinement tests smaller perturbations while preserving the last successful image.

**$L_2$.** Local additive mutations did not work consistently across the supplied images. I instead redistribute perturbation energy between spatial regions, using structured positive and negative patterns. The search initially explores candidates under a global energy budget and contracts that budget after success. It then evolves feasible directions near the classification boundary and tests additional scalar reductions. The current implementation preserves global exploration for at least 512 proposal generations and three successful stages before switching to boundary refinement. This was introduced after an earlier version switched too early on some new images.

**$L_0$.** The main difficulty is not only finding a successful perturbation, but also determining which changed elements are actually needed. I first evolve sparse attacks, then attempt to remove complete channel or spatial groups. Failed groups are divided into smaller groups, and remaining single elements are checked again after other deletions, since an earlier failed deletion may become possible later. When the search converges, independent restarts generate additional candidates. A further GA searches smaller combinations of the changed elements found in successful parents. Each stage keeps the strongest verified attack. This combines genetic discovery with explicit support reduction rather than requiring the initial population to be minimal.

**Perlin fog (bonus).** I represent a fog candidate as a list of octave and magnitude pairs. The GA initially searches mixtures over a fixed set of octave layers, using crossover and mutation of their relative weights. A scalar search then reduces the total magnitude while retaining misclassification. The remaining time is used to evolve additional integer octaves. The final genotype is checked again through the supplied Perlin renderer, rather than relying only on the faster search representation.

## 3. Results on the Supplied Images

The table reports the lowest verified distance saved for each supplied image. Each result is a valid adversarial example under the assignment's metric. The files were selected from several experiments and seeds; these values therefore describe the best saved attacks, not the outcome of one run of the final search function.

| Image | $L_\infty$ | Squared $L_2$ | $L_0$ | Fog magnitude |
|:--|--:|--:|--:|--:|
| knot | 0.163680 | 386.170 | 109 | 0.912109 |
| mink | 0.067076 | 235.239 | 31 | 0.413872 |
| orca | 0.162440 | 940.956 | 82 | 0.501953 |
| sock | 0.042834 | 131.672 | 22 | 0.123452 |
| tank | 0.103578 | 595.750 | 84 | 0.903415 |

The largest improvement came from reducing the support of successful $L_0$ attacks. For tank, the first 900-query GA result changed 2,240 channel elements, whereas the best saved result changes 84. Controlled comparisons at two seeds also favored hierarchical pruning over the original GA in all ten image-seed pairs. The comparison of 2,240 and 84 uses different searches and budgets, so it measures the improvement of the saved result rather than equal-budget efficiency.

For the bonus task, the best saved fog attacks each use a single octave: knot (24, 0.912109), mink (108, 0.413872), orca (24, 0.501953), sock (65, 0.123452), and tank (60, 0.903415). The broader octave search improved the saved fog magnitudes for mink, sock, and tank. It did not establish that mixtures are generally worse; these are the best candidates found for the five supplied images.

## 4. Generalization Experiments

The assignment also evaluates the submitted functions on images not supplied in advance. I therefore compared search strategies on separate images instead of selecting algorithms only from the five required examples.

An initial four-image confirmation set favored the revised $L_2$ method on all four images at 900 model calls. Its geometric-mean ratio of squared distance to the previous global-only GA was 0.788. However, this small set included uncertain or semantically inconsistent clean predictions. I then used a labelled Imagenette development set with ten different classes, keeping only images whose clean predictions matched their labels. On these ten images, the same revised method won five comparisons and lost five, with a geometric-mean distance ratio of 1.040. The earlier result did not establish general superiority.

I also tested two methods that alternated global and local search. The first candidate lost all ten 900-call comparisons against the current implementation, with a geometric-mean ratio of 1.176. A second candidate preserved the global search state across switches, but won only two of ten comparisons and had a ratio of 1.050. Under the actual 165-second budget, neither passed the predefined selection criteria. I retained the existing implementation. The ten reserved Imagenette confirmation images were not used to tune or select these rejected candidates.

These experiments show why a useful local improvement is not necessarily a better search policy. Global exploration can still be productive when local refinement begins, but returning to global exploration also consumes evaluations that might have improved a feasible candidate. The choice depends on the result after a fixed computation budget, rather than on whether either operator can make progress in isolation.

## 5. Final Selection

The final `epsilon_ball.py` and `perlin_fog.py` preserve the required interfaces and use the same general search procedures on new inputs. All 15 required raw tensors and the five bonus fog cases passed independent verification, including reloading and regeneration of the fog images. The corresponding `get_viz_for` classification plots are preserved with the verified attacks.

The strongest evidence of improvement is the reduction of unnecessary $L_0$ changes and the better use of the available fog search budget. The $L_2$ experiments also found cases where a method improved the supplied images but became worse on other inputs. I therefore kept the best verified implementation instead of replacing it with a new strategy based on selected examples. Performance on the instructor's hidden images remains to be measured by the grader.
