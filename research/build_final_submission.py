"""Build the minimal README deliverable overlay and its evidence-based report.

Build dependencies are intentionally outside the submitted attack modules.
The authoritative figures/tensors are copied unchanged from submission/.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

STEMS = ('knot', 'mink', 'orca', 'sock', 'tank')
NORMS = ('linf', 'l2', 'l0', 'perlin')
LABELS = {'linf':'L∞', 'l2':'Squared L2', 'l0':'L0 elements', 'perlin':'Fog magnitude'}
REPORT = 'CS454_Assignment3_Report.pdf'
ACCENT = colors.HexColor('#14566b')
INK = colors.HexColor('#162632')
MUTED = colors.HexColor('#51616c')
PALE = colors.HexColor('#edf4f6')
PAGE_W, PAGE_H = A4
MARGIN = 46
WIDTH = PAGE_W-2*MARGIN


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fonts():
    folder = Path('/usr/share/fonts/truetype/lato')
    for name, file in [('Body','Lato-Regular.ttf'),('BodyBold','Lato-Bold.ttf'),
                       ('BodyItalic','Lato-Italic.ttf'),('BodyBoldItalic','Lato-BoldItalic.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(folder/file)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='BodyItalic',
                                  boldItalic='BodyBoldItalic')
    pdfmetrics.registerFont(TTFont('Mono','/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
    pdfmetrics.registerFont(TTFont('Equation','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))


def styles():
    return {
        'body':ParagraphStyle('body',fontName='Body',fontSize=10.2,leading=14.2,
                              textColor=INK,spaceAfter=8),
        'small':ParagraphStyle('small',fontName='Body',fontSize=8.5,leading=11.5,
                               textColor=MUTED,spaceAfter=6),
        'table':ParagraphStyle('table',fontName='Body',fontSize=8.5,leading=11.2,textColor=INK),
        'th':ParagraphStyle('th',fontName='BodyBold',fontSize=8.4,leading=11.2,textColor=colors.white),
        'title':ParagraphStyle('title',fontName='BodyBold',fontSize=24,leading=29,
                               textColor=INK,spaceAfter=12),
        'h1':ParagraphStyle('h1',fontName='BodyBold',fontSize=16,leading=21,
                            textColor=ACCENT,spaceAfter=12),
        'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=11.4,leading=15,
                            textColor=INK,spaceBefore=6,spaceAfter=6),
        'eq':ParagraphStyle('eq',fontName='Equation',fontSize=10,leading=16,
                            textColor=INK,alignment=TA_CENTER,spaceAfter=9),
        'code':ParagraphStyle('code',fontName='Mono',fontSize=7.8,leading=11.5,
                              textColor=INK,backColor=PALE,borderPadding=9,spaceAfter=11),
        'caption':ParagraphStyle('caption',fontName='Body',fontSize=8.7,leading=11.7,
                                 textColor=MUTED,spaceAfter=9),
    }


class Report:
    def __init__(self, output, work):
        self.output, self.work = output, work
        self.style = styles()
        self.story = []
        self.text_parts = []
        self.verification = read_csv(ROOT/'submission/verification.csv')
        self.by_pair = {(r['image'],r['metric']):r for r in self.verification}
        self.names = ast.literal_eval((ROOT/'category_names.txt').read_text())
        self.general = json.loads((ROOT/'runs/imagenette-allnorm-confirmation-validation-20261123/report/summary.json').read_text())
        self.wide = read_csv(ROOT/'runs/imagenette-allnorm-confirmation-validation-20261123/execution-report-v2/image_norm_wide.csv')
        self.figure_files = []

    def p(self, text, style='body'):
        self.story.append(Paragraph(text, self.style[style]))
        self.text_parts.append(text)

    def h(self, text):
        self.p(text,'h2')

    def page(self, title):
        if self.story:
            self.story.append(PageBreak())
        self.p(title,'h1')

    def table(self, rows, widths=None, small=False):
        data=[]
        for index,row in enumerate(rows):
            data.append([Paragraph(str(value), self.style['th' if index==0 else 'table']) for value in row])
            self.text_parts.append(' | '.join(map(str,row)))
        table=Table(data,colWidths=widths or [WIDTH/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),ACCENT),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),5 if not small else 3),
            ('BOTTOMPADDING',(0,0),(-1,-1),5 if not small else 3),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
            ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#c2d0d6')),
        ]))
        self.story += [table,Spacer(1,9)]

    def short(self, index):
        return self.names[int(index)].split(',')[0]

    def build_text(self):
        self.p('CS454 / ASSIGNMENT 3','small')
        self.p('Genetic Algorithms for<br/>Adversarial Testing','title')
        self.p('Implementation, development evidence and final evaluation<br/>23 November 2026','small')
        self.h('Executive summary')
        self.p('The project implements gradient-free genetic searches for a fixed pretrained VGG-11, '
               'using different perturbation representations for L∞, L2 and L0 and an optional Perlin-fog genotype. '
               'A protected successful archive separates search fitness from the final objective. '
               'The submitted bundle contains 15 required adversarial tensors and five bonus fog cases, '
               'their original classification visualizations, legal fog expressions and the two required implementations.')
        self.p('The strongest verified source was retained after two later global/local scheduling designs failed selection. '
               'A final, frozen Imagenette assessment succeeded on all 10 inputs for each of four attacks (40/40), '
               'including independent one-thread verification. This supports input transfer for the tested configuration; '
               'it neither establishes universal generalization nor proves that the perturbations are globally minimal.')
        self.h('1. Task, baseline and scoring contract')
        self.p('The supplied ZIP and README define the interfaces and scoring [1]. Both original attack modules '
               'contain unimplemented entry points. Two independent clean CPU runs reproduced all five expected classes; '
               'that established the reference predictions, not an adversarial optimization baseline. '
               'The final code preserves the instructor utilities, grader and the three supplied fog-rendering functions.')
        self.p('Let x be the normalized 1 × 3 × 224 × 224 tensor returned by the unchanged loader. '
               'That loader resizes the image and applies the VGG weight transforms; distances are not RGB 8-bit pixel distances. '
               'The attack infers y = argmax p(x), rather than looking up a filename or label. Candidate images are clipped '
               'to [−3, 3], matching the model wrapper. A successful return must change the actual argmax.')
        self.p('D∞(δ) = max<sub>i</sub> |δ<sub>i</sub>| &nbsp;&nbsp; '
               'D2(δ) = Σ<sub>i</sub> δ<sub>i</sub>² &nbsp;&nbsp; '
               'D0(δ) = Σ<sub>i</sub> 1[δ<sub>i</sub> ≠ 0]','eq')
        self.p('<b>Two important grading details:</b> the supplied “L2” helper reports the sum of squares, '
               'not its square root; L0 counts changed channel elements, not RGB pixels. '
               'The five seed reference IDs are knot 616, mink 357, orca 148, sock 806 and tank 847.')
        self.p('Adversarial-example research motivates small, deliberate perturbations [2,3]; sparse evolutionary '
               'attacks motivate support search [4], and weather transformations motivate the optional fog task [5]. '
               'This work does not implement or benchmark those papers’ algorithms; all measured comparisons below '
               'are between this project’s own GA variants.','small')

        self.page('2. Shared search and dense perturbations')
        self.h('2.1 Fitness, genetic operators and protected feasibility')
        self.p('The fitness is the untargeted log-probability margin below, with probabilities floored at 10<super>−35</super> '
               'for numerical stability. A negative margin means a rival outranks the reference. '
               'Archive acceptance still checks the actual argmax and authoritative distance, so fitness ties '
               'or later infeasible candidates cannot erase a valid result.')
        self.p('m(x′) = log p<sub>y</sub>(x′) − max<sub>k ≠ y</sub> log p<sub>k</sub>(x′)','eq')
        self.p('Dense search uses four truncation-selected elites. The best parent is chosen with probability 0.8; '
               'otherwise a parent is drawn from the elite pool. Spatial crossover occurs periodically and mutation '
               'introduces new structure. Infeasible elites may guide search at a harder radius, while a separate '
               'archive retains the lowest-distance classifier-verified image. Scalar bisection and explicit deletion '
               'are deterministic refinements around the GA, not genetic crossover or gradient estimates.')
        self.h('2.2 L∞: signed blocks and radius contraction')
        self.p('The chromosome is a full perturbation tensor bounded by its maximum component. Initial vertical '
               'sign stripes provide inexpensive diversity. A mutation replaces a randomly positioned square '
               'with independently signed channel values; periodic crossover copies another elite’s block. '
               'The initial radius is 0.4. After success, the target contracts by 0.8; repeated failure before '
               'discovery can expand it by 1.6. The spatial fraction decreases within a stage, permitting finer changes. '
               'Up to 12 final scalar-bisection evaluations reduce the successful direction without losing feasibility.')
        self.h('2.3 L2: structured energy and delayed boundary evolution')
        self.p('Early additive color-block mutations were unreliable. The selected representation initializes smooth '
               'positive/negative lobes over spatial tiles. Mutation recovers energy from two windows, handles their '
               'overlap once, and reallocates energy into a perturbed direction while projecting to the allowed '
               'Euclidean radius. The search starts at radius 30, with 0.8 contractions after success. '
               'The reported score is that radius’s squared-distance counterpart.')
        self.p('The final policy keeps global exploration for at least <b>512 main proposals and three successful stages</b>. '
               'It then evolves feasible directions near the boundary: target radius 0.99√D2, a 0.25 blend of parent '
               'and mutated direction, and up to three extra 0.98 scalar contractions after an accepted direction. '
               'Every probe is a charged model query; a failed probe never becomes the next feasible parent. '
               'Stagnation reduces the angular blend and can relax an overly hard boundary target.')
        self.table([['Control','Final default / role'],['Norm budget','165 s, four Torch threads; configurable hard maximum 170 s'],
                    ['Cleanup reserve','Observed-forward-time slack and reserved final refinement calls'],
                    ['Randomness','Default seed 454; separate fixed seeds used in recorded comparisons'],
                    ['Control route','CS454_L2_SEARCH=original selects the preserved global-only GA']], [126,WIDTH-126])

        self.page('3. Sparse changes and the fog genotype')
        self.h('3.1 L0: discovery, hierarchy, restarts and support crossover')
        self.p('Sparse discovery evolves large-valued channel-element changes. Once an adversary exists, '
               'discovery is bounded using a 0.55 fraction and a 900-call reference allocation, retaining the full '
               'fallback if no success has been found. This avoids spending most of the budget on a marginally '
               'smaller but difficult-to-prune discovery parent.')
        self.p('Refinement first tests whole-channel and spatial-group deletion. A failed group is split along its '
               'widest coordinate; accepted deletions preserve misclassification. Singleton deletions are revisited '
               'after other elements disappear, because classifier interactions can make an earlier failed deletion '
               'possible later. “Deletion minimal” therefore refers to tested removals at the current endpoint values, '
               'not a proof of the globally smallest support.')
        self.p('Protected restarts create independent parents. A further GA searches fixed-cardinality subsets of '
               'their combined loci: the target is one element smaller than the stronger parent. Four elites retain '
               'low-margin subsets, locus-exchange mutations and crossover/repair preserve target cardinality, and '
               'accepted children lower the target again. Overlapping loci inherit the stronger parent’s endpoint '
               'rather than adding amplitudes. The exact globally best feasible tensor survives failed restarts or offspring.')
        self.h('3.2 Fog: mixtures followed by integer-octave evolution')
        self.p('A genotype g is a list of positive-integer octaves and positive finite magnitudes, with at most '
               '50 pairs. It produces the instructor’s deterministic Perlin fields, with generator seed 1. '
               'Opacity is clipped to [0,1] before blending toward the supplied normalized white color. '
               'The grading cost is only the sum of magnitudes; it is not an Lp image distance.')
        self.p('F<sub>g</sub> = Σ<sub>j</sub> a<sub>j</sub>P<sub>oⱼ</sub>, &nbsp; '
               'α = clip(F<sub>g</sub>, 0, 1), &nbsp; x′ = (1 − α)x + αw, &nbsp; '
               'D<sub>fog</sub>(g) = Σ<sub>j</sub> a<sub>j</sub>','eq')
        self.p('The first GA uses 16 candidates over octave layers {1,2,3,4,6,8,12,24,32,48,64,96}. '
               'Pure layers and random mixtures seed the population; successful directions receive scalar boundary '
               'searches. Four parents are selected, crossover mixes their weights, and Gaussian mutations alternate '
               'broad and fine scales. The mixture phase has an 80-generation allowance and a stagnation stop. '
               'The remaining time evolves integer octaves 1–128 for up to 24 additional generations, using arithmetic '
               'crossover, local/global mutation and scalar magnitude reduction.')
        self.p('Layers are cached only within a search and model evaluations use batches of at most four. '
               'Fast search arithmetic is never substituted for the grading renderer: the final genotype is '
               'regenerated and classified exactly, with a protected verified mixture fallback. '
               'The common fog target is 270 s, below the 300 s limit; the final confirmation also timed '
               'mandatory exact rendering and classification. This is the assignment’s fog-like perturbation, '
               'not a calibrated atmospheric visibility model.')

        self.page('4. Development decisions and rejected approaches')
        self.p('Development followed a verified-baseline → hypothesis → controlled comparison → retention/rejection '
               'cycle. Equal query ceilings are distinguished from equal actual calls; equal time ceilings are '
               'distinguished from equal runtimes. The original starter’s failure sentinel 999999999 is not a measured '
               'perturbation score, and clean classification accuracy is not attack success.')
        self.table([
            ['Investigation','Measured evidence','Interpretation'],
            ['Dense representation','At 450-call ceilings, additive L2 succeeds 2/5; energy variant 5/5.','Energy adopted; a 12-call failure-accounting fix is a small confound.'],
            ['Hierarchical L0 pruning','Tank: 2240 → 99 at the 900-call ceiling. Ten image/seed pairs favor refinement.','Prune successful support rather than rely on random discovery alone.'],
            ['Compact rectangle L0','Tank: 6228 versus 2240 at the same seed / 900-call ceiling.','Rejected; a compact genotype did not imply better sparse search.'],
            ['Discovery caching','One tank variant regresses 99 → 130 after changing discovery allocation.','Fewer repeated calls can change trajectory; caching is not automatically a quality gain.'],
            ['Parental support GA','Seed 455, 1800-call ceilings: mink 41 → 31; sock 46 → 43.','Uses otherwise idle allowance; actual candidate work is larger.'],
            ['Full-prefix fog extension','90-s ceiling: mink .55664 → .45596; sock .16992 → .12345; tank .97656 → .90342.','Spends unused convergence time; not equal actual runtime or per-query evidence.'],
        ],[105,200,WIDTH-305],small=True)
        self.p('The first 900-call L0 tank result of 2240 and the final saved result of 84 use different methods and '
               'budgets. They demonstrate the evolution of the saved artifact, not a controlled efficiency ratio. '
               'Similarly, some final fog artifacts originated in 90-second runs, while the final function permits '
               '270 seconds. Successful phenotype protection was retained throughout.')
        self.h('Engineering checks that changed decisions')
        self.p('A noncontiguous input initially made flattened indexed writes modify a copy instead of the candidate; '
               'the implementation now makes the input contiguous. Export tests also caught a one-ulp endpoint '
               'change when reconstructing x + (image − x); the archive now preserves the exact evaluated tensor. '
               'These correctness checks matter near a classification boundary. Shape, dtype, finite values, '
               'thread restoration, randomness, archive protection and exact serialization are tested separately '
               'from real-model performance.')
        self.p('Later experiments showed that evidence from the five supplied images was insufficient for policy '
               'selection. The next page documents why the selected delayed-boundary method was retained despite '
               'mixed labelled development results, and why two scheduling alternatives were rejected.')

        self.page('5. Generalization-driven selection of L2')
        self.p('The first external bank used eight fixed scikit-image fixtures: four development and four '
               'confirmation inputs. It used clean argmax as the attack reference without semantic eligibility filtering. '
               'Some inputs had low-confidence or semantically inconsistent clean predictions. '
               'The following ratios are candidate/control <b>squared-distance</b> ratios; lower is better.')
        self.table([
            ['Comparison','Protocol','Wins','Geom. ratio'],
            ['Early boundary switch / global-only','4 fixture development images, 900 calls','1/4','1.400'],
            ['512-proposal policy / global-only','4 fixture development images, 900 calls','4/4','0.776'],
            ['Frozen selected policy / global-only','4 fixture confirmation images, seed 455, 900 calls','4/4','0.788'],
            ['Same frozen policy / global-only','2 predeclared fixture confirmation images, 165 s','2/2','0.599'],
            ['Current policy / global-only','10 labelled development images, seed 456, 900 calls','5/10','1.040'],
            ['First reversible windows / current','Same 10 labelled development images, 900 calls','0/10','1.176'],
            ['State-preserving windows / current','Same 10 labelled development images, 900 calls','2/10','1.050'],
        ],[184,190,40,WIDTH-414],small=True)
        self.p('The 512-proposal requirement corrected premature local refinement on the first bank. '
               'However, the later ten-image labelled audit had a worst current/global-only ratio of 1.638. '
               'Thus the earlier confirmation result supported that small bank, not universal superiority. '
               'Fixed-query controls were reused only when source, seed, input and settings matched; an independent '
               'recovery replay reproduced a complete 900-call trajectory and final tensor exactly.')
        self.h('Reversible allocation did not pass the actual-time gate')
        self.p('The first window design alternated 96-call global/local windows using realized progress. '
               'A second design retained pending global elites, absolute targets, proposal clocks and separate RNG '
               'streams. The latter produced useful global gains on some inputs, but progress within an operator '
               'did not necessarily offset the local refinement it displaced.')
        self.table([
            ['Candidate vs. current','165-s development pairs','Geom. ratio','Worst ratio'],
            ['First reversible windows','0 wins / 2 pairs','1.122','1.143'],
            ['Preserved-state windows','1 win / 2 pairs','1.110','1.244'],
        ],[184,149,83,WIDTH-416])
        self.p('Both failed the frozen full-time selection gate (aggregate ≤0.95, worst ≤1.05, no success regression). '
               'They were not promoted or tuned using confirmation. Fresh controls mattered: the gas-pump current '
               'control changed from 778.135 at 2096 calls to 635.221 at 2675 calls in separate 165-second windows. '
               'Reusing the earlier wall-time denominator would have understated the preserved-state regression. '
               'The production source was therefore fixed before the final all-norm confirmation.')

        self.page('6. Submitted results on the five assignment seeds')
        self.p('Table 1 reports the lowest verified saved artifact for each case. All 15 norm cases and five bonus '
               'fog cases change the expected class under the supplied utilities. Distances are recomputed from '
               'the stored raw tensors; fog cost and classification are computed from exact JSON regeneration. '
               '<b>These are a multi-run artifact portfolio, not a uniform run of the final default function.</b> '
               'Different seeds, methods and query/time budgets contributed to selection.')
        rows=[['Image','L∞','Squared L2','L0','Fog magnitude']]
        for stem in STEMS:
            r={norm:self.by_pair[stem,norm] for norm in NORMS}
            rows.append([stem,f"{float(r['linf']['distance']):.6f}",f"{float(r['l2']['distance']):.6f}",
                         r['l0']['distance'],f"{float(r['perlin']['distance']):.9f}"])
        self.table(rows,[70,85,115,60,WIDTH-330])
        self.p('Table 1. Saved grading distances. The full values remain in the raw tensors and gene JSON; '
               'minor final-digit L2 differences can occur between parallel reduction thread counts.','caption')
        self.h('Observed class changes for every submitted case')
        rows=[['Image / reference','Attack','Predicted class (ID)','Result']]
        for stem in STEMS:
            for index,norm in enumerate(NORMS):
                r=self.by_pair[stem,norm]
                rows.append([f"{stem} ({r['reference_class']})" if index==0 else '',LABELS[norm],
                             f"{escape(self.short(r['predicted_class']))} ({r['predicted_class']})",'Pass'])
        self.table(rows,[118,96,WIDTH-260,46],small=True)
        self.p('The clean reference names are knot, mink, killer whale, sock and tank. '
               'Figures 1–20 reproduce the actual get_viz_for plots for these artifacts, including the image '
               'and top-ten class probabilities. A rival need only exceed the reference; success does not require '
               'a top-class probability above 0.5.','small')

        self.page('7. Exact bonus expressions and interpretation')
        self.p('Each submitted bonus winner uses one octave. The expressions below are the complete saved '
               'genotypes, not rounded descriptions; magnitude is also the exact grading cost for a single pair. '
               'All octave values are positive integers and all magnitudes are positive finite numbers. '
               'The instructor’s unchanged generator supplies the Perlin realization, so no extra noise-image '
               'or random-state file is needed.')
        rows=[['Image','Complete genotype [(octave, magnitude)]','Predicted class']]
        for stem in STEMS:
            genes=json.loads((ROOT/f'submission/{stem}_perlin.json').read_text())
            r=self.by_pair[stem,'perlin']
            rows.append([stem,'<font name="Mono" size="7.7">'+escape(repr([tuple(g) for g in genes]))+'</font>',
                         f"{escape(self.short(r['predicted_class']))} ({r['predicted_class']})"])
        self.table(rows,[62,279,WIDTH-341])
        self.p('The broader integer-octave stage found winners outside the initial layer set for mink (108), '
               'sock (65) and tank (60). The best saved magnitudes improved from 0.556640625 to 0.413872064 '
               'for mink, 0.169921875 to 0.123451920 for sock, and 0.976562500 to 0.903415341 for tank. '
               'Knot and orca retained their earlier verified octave-24 artifacts.')
        self.p('Single-octave winners do not show that mixtures are generally inferior. The implementation '
               'retains the complete mixture search and its verified fallback, then tests a wider octave family '
               'using remaining time. Likewise, initial pure-octave screening, later genetic evolution and scalar '
               'boundary search all contribute; the final objective cannot be attributed to mutation alone.')
        self.h('What the saved portfolio does and does not establish')
        self.p('Every submitted file is directly checkable without rerunning an optimizer. The exact tensor is '
               'preserved; filenames follow the README and all expected class changes are verified. '
               'However, a fresh call to the final code on a different CPU need not reproduce the minimum saved '
               'distance: wall stopping, seed and search trajectory vary, and older verified searches contributed '
               'some files. The next section therefore reports a separate, uniform-configuration assessment of '
               'the final source on previously reserved inputs.')
        self.h('Minimal submission layout')
        self.p('This directory is an <b>overlay for the original cs454-assignment3-adversarial.zip skeleton</b>. '
               'Copy all files into the extracted cs454-adversarial project root, replacing only the two student '
               'implementation modules. Preserve the instructor’s util.py, grade.py, category_names.txt, dependency '
               'lists and images. Run commands from that project root, because the supplied utility reads the '
               'class-name file using a relative path.')
        self.table([['Files','Count'],['epsilon_ball.py and perlin_fog.py','2'],
                    ['This PDF report','1'],['raw_&lt;image&gt;_&lt;norm&gt;.png (three norms + perlin)','20'],
                    ['&lt;image&gt;_&lt;norm&gt;.png classification visualizations','20'],
                    ['&lt;image&gt;_perlin.json complete genotypes','5'],['Total: only required work and attempted bonus deliverables','48']],
                   [WIDTH-65,65])
        self.p('The files named raw_*.png are PyTorch tensor serializations written by the instructor helper, '
               'not ordinary bitmap PNGs. The provided reference files, datasets, model checkpoint, caches, '
               'research variants and logs are intentionally not duplicated in this minimal overlay.','small')

        self.page('8. Frozen all-norm Imagenette confirmation')
        self.p('A separate bank used the official Imagenette2-160 validation archive [6]. Per-class candidate pools '
               'were fixed by hashes before clean forwards. The first clean-correct image in each role/class was '
               'selected, without confidence or attack filtering: 20 images, ten development and ten confirmation, '
               'with exclusions recorded. Clean correctness is an eligibility condition, not a classifier-accuracy estimate.')
        self.p('The final source, seed 457, settings, image hashes, environment and 40-case order were frozen before '
               'confirmation attacks. Each norm had a 165-second target and fog 270 seconds, with four Torch '
               'threads and no query ceiling. Following a request to accelerate execution, 11 completed serial '
               'cases were preserved and the remaining 29 used two workers pinned to disjoint four-core groups '
               '(logical CPUs 0,4,8,12 and 2,6,10,14). Shared memory bandwidth, package power and host scheduling '
               'remain uncontrolled. Execution mode is recorded per case; no equal-throughput or comparative '
               'algorithm-improvement claim is made from the mixed conditions.')
        rows=[['Confirmation class','L∞','Squared L2','L0','Fog']]
        for r in self.wide:
            values=[r['expected_name'].split(',')[0]]
            for norm in NORMS:
                value=float(r[f'{norm}_measured_distance'])
                values.append(str(int(value)) if norm=='l0' else f'{value:.6g}')
            rows.append(values)
        self.table(rows,[142,88,99,48,WIDTH-377],small=True)
        rows=[['Attack','Success','Median distance','Max grading seconds']]
        for norm in NORMS:
            r=self.general['per_norm'][norm]
            rows.append([LABELS[norm],'10 / 10',f"{r['successful_distance_median']:.6g}",f"{r['grading_runtime_max']:.3f}"])
        self.table(rows,[122,67,145,WIDTH-334])
        self.p('All 40 saved tensors passed independent one-thread reclassification and authoritative metric '
               'checks; every predicted class also matched the original four-thread record. Fog was regenerated '
               'from each JSON. Squared-L2 verification permits 10<super>−6</super> relative numeric tolerance '
               'across thread reductions; L∞ and L0 match exactly. Grading time includes attack plus required '
               'distance/rendering and classification, excluding export and validation. No 180/300-second '
               'grading timeout or 165/270-second optimizer overrun occurred.','small')

        self.page('9. Conclusions, limits and reproducibility')
        self.h('9.1 Conclusions supported by the measurements')
        self.p('Different metrics favored different representations: signed blocks control the largest component; '
               'structured energy plus feasible-direction evolution distributes L2 changes; explicit support '
               'deletion and parental-locus evolution reduce L0; octave/magnitude genes expose a compact fog family. '
               'The largest supplied-image sparse improvements came from removing redundant successful support. '
               'The fog extension benefited from a wider octave range and time the earlier stagnation stop left unused.')
        self.p('The failed compact rectangle, caching and reversible-window experiments are part of the result. '
               'Representation simplicity, fewer repeated calls, or an operator that sometimes improves the archive '
               'did not guarantee a better final distance. The clean baseline, saved-attack portfolio, fixed-call '
               'comparisons and actual-time confirmation answer different questions and were kept separate.')
        self.h('9.2 Limits and next investigations')
        self.p('Forty successful attacks cover only ten correlated inputs, one photo per class and one seed. '
               'Imagenette is ImageNet-derived and may overlap classifier pretraining. The 40 cases are an absolute '
               'assessment of one source, not a paired comparator or an estimate over all ImageNet classes. '
               'French horn remained costly: squared L2 4039.34 and L0 177; no global-minimality claim is justified. '
               'The confirmation bank is now consumed and cannot be treated as fresh independent validation after tuning.')
        self.p('Further development should test a distinct L2 spatial-coefficient representation and L0 support-space '
               'exhaustion/neighbor expansion, with matched actual-time controls and another seed. Such changes '
               'belong on development data and require a new independent bank before another generalization claim. '
               'The instructor’s true hidden evaluation remains unknown.')
        self.h('9.3 Tested execution environment and replay')
        self.p('Measured environment: Python 3.14.4; torch 2.15.0+cpu; torchvision 0.29.2+cpu; NumPy 2.5.4; '
               'Pillow 13.0.0; matplotlib 3.11.2; perlin-noise 1.13. CPU: AMD Ryzen AI 7 350, Linux/WSL2. '
               'Default search seed is 454; final confirmation uses 457. There is no GPU or gradient-based optimization.')
        self.p('After applying the 48-file overlay to the original skeleton:','small')
        self.p('pip install -r torch_cpu.txt<br/>pip install -r requirements.txt<br/>'
               'python grade.py -n final -i all -l all','code')
        self.p('The first model import downloads the official VGG-11 checkpoint if no cache exists. '
               'The full command reruns searches and may take several minutes per case; it checks the current '
               'functions rather than reproducing a multi-run artifact selection. CS454_SEED, CS454_THREADS '
               'and CS454_SECONDS expose norm experiment settings; CS454_FOG_SECONDS controls fog. '
               'Use the submitted defaults for the reported source configuration.','small')
        self.p('The final combined contracts cover 93 distinct checks. A successful combined invocation executed '
               '96 tests because one three-test module appeared twice; no real attack was repeated. '
               'These synthetic checks complement, but do not replace, actual classifier validation. '
               'The final-package audit also loads the 20 submitted raw results with the original ZIP utilities.','small')

        self.page('10. References and artifact verification')
        refs=[
            ('1','CS454 assignment materials. README.md, util.py, grade.py and starter modules in '
             'cs454-assignment3-adversarial.zip (provided course skeleton). The local README has the same '
             'requirements; its only text difference is a trailing “# CS454”.',None),
            ('2','I. J. Goodfellow, J. Shlens and C. Szegedy. Explaining and Harnessing Adversarial Examples. '
             'arXiv:1412.6572, 2014; revised 2015.','https://arxiv.org/abs/1412.6572'),
            ('3','N. Carlini and D. Wagner. Towards Evaluating the Robustness of Neural Networks. '
             'arXiv:1608.04644, 2016; revised 2017.','https://arxiv.org/abs/1608.04644'),
            ('4','J. Su, D. V. Vargas and K. Sakurai. One pixel attack for fooling deep neural networks. '
             'arXiv:1710.08864, 2017; IEEE Transactions on Evolutionary Computation, 2019.',
             'https://arxiv.org/abs/1710.08864'),
            ('5','Y. Tian, K. Pei, S. Jana and B. Ray. DeepTest: Automated Testing of '
             'Deep-Neural-Network-driven Autonomous Cars. arXiv:1708.08559, 2017; revised 2018.',
             'https://arxiv.org/abs/1708.08559'),
            ('6','fast.ai. Imagenette: a ten-class ImageNet subset. Experiments use the official '
             'imagenette2-160.tgz validation archive, not a claim of classifier-training-unseen data.',
             'https://github.com/fastai/imagenette'),
        ]
        for number, text, url in refs:
            self.p(f'[{number}] '+escape(text)+(f'<br/><link href="{url}" color="#14566b">{url}</link>' if url else ''),'small')
        self.h('Direct check of an included raw tensor')
        self.p('The supplied helpers can validate a saved result without rerunning its GA:','small')
        self.p('import torch<br/>from util import *<br/>torch.set_num_threads(1)<br/>'
               'with torch.inference_mode():<br/>'
               '&nbsp;&nbsp;x = get_image_tensor_from("images/knot.jpg")<br/>'
               '&nbsp;&nbsp;z = torch.load("raw_knot_l2.png",<br/>'
               '&nbsp;&nbsp;&nbsp;&nbsp;map_location="cpu", weights_only=True)<br/>'
               '&nbsp;&nbsp;print(get_L2_norm_for(z - x))<br/>'
               '&nbsp;&nbsp;print(get_model_most_likely_for(z))','code')
        self.p('Expected knot result: squared L2 ≈386.169647 and predicted class327 (starfish), different from '
               'the reference616. For fog, load &lt;image&gt;_perlin.json, call genotype2phenotype followed by '
               'add_noise_to_image, compare the resulting prediction, and sum the magnitudes. '
               'All 20 raw filenames and 20 visualization filenames appear beside their figures on the following pages.','small')
        self.h('Integrity anchors')
        for name in ('epsilon_ball.py','perlin_fog.py'):
            self.p(escape(name)+':<br/><font name="Mono" size="7.0">'+sha(ROOT/name)+'</font>','small')
        self.p('VGG-11 checkpoint SHA-256:<br/><font name="Mono" size="7.0">'
               '8a7190461cb581dbcbd2c3b58f43cf12a0bca3ccf92898f867e4a9e9b898c464</font>','small')

    def build_figures(self):
        index=1
        for stem in STEMS:
            reference=self.by_pair[stem,'linf']['reference_class']
            for pair_id, pair in enumerate((('linf','l2'),('l0','perlin')),1):
                self.page(f'Visual results: {stem} ({pair_id}/2)')
                self.p(f'Clean reference: {escape(self.names[int(reference)])} [{reference}]. '
                       'Plots are the unmodified instructor get_viz_for outputs for the submitted raw tensors.', 'small')
                for norm in pair:
                    row=self.by_pair[stem,norm]
                    image_file=ROOT/f'submission/{stem}_{norm}.png'
                    self.figure_files.append(str(image_file.relative_to(ROOT)))
                    im=PILImage.open(image_file)
                    width=WIDTH-8
                    height=width*im.height/im.width
                    title=Paragraph(f'<b>Figure {index}. {stem} — {LABELS[norm]}</b>',self.style['caption'])
                    picture=Image(str(image_file),width=width,height=height,hAlign='CENTER')
                    distance=str(int(float(row['distance']))) if norm=='l0' else f"{float(row['distance']):.9g}"
                    caption=Paragraph(f'Distance: <b>{distance}</b>. Prediction: '
                        f'{escape(self.names[int(row["predicted_class"])])} [{row["predicted_class"]}].<br/>'
                        f'Visualization: <font name="Mono" size="7.7">{stem}_{norm}.png</font>; '
                        f'raw tensor: <font name="Mono" size="7.7">raw_{stem}_{norm}.png</font>.',
                        self.style['caption'])
                    self.story.append(KeepTogether([title,picture,Spacer(1,4),caption,Spacer(1,7)]))
                    self.text_parts.append(f'Figure {index}: {stem}_{norm}.png; raw_{stem}_{norm}.png; '
                                           f'distance {distance}; predicted {row["predicted_class"]}')
                    index+=1

    def draw_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(MUTED);canvas.setFont('BodyBold',8)
        canvas.drawString(MARGIN,PAGE_H-28,'CS454 / ASSIGNMENT 3')
        canvas.setFont('Body',8)
        canvas.drawRightString(PAGE_W-MARGIN,PAGE_H-28,'Genetic Algorithms for Adversarial Testing')
        canvas.setStrokeColor(colors.HexColor('#c6d4d9'));canvas.setLineWidth(.5)
        canvas.line(MARGIN,PAGE_H-37,PAGE_W-MARGIN,PAGE_H-37)
        canvas.line(MARGIN,30,PAGE_W-MARGIN,30)
        canvas.setFont('Body',7.5)
        canvas.drawString(MARGIN,19,'Final implementation and verified artifacts • 23 November 2026')
        canvas.drawRightString(PAGE_W-MARGIN,19,str(doc.page))
        canvas.restoreState()

    def build(self):
        self.build_text();self.build_figures()
        target=self.work/REPORT
        doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
            topMargin=54,bottomMargin=43,title='CS454 Assignment 3 — Genetic Algorithms for Adversarial Testing',
            author='CS454 Assignment 3',subject='Implementation, development analysis, artifacts and confirmation results',
            pageCompression=1)
        doc.build(self.story,onFirstPage=self.draw_page,onLaterPages=self.draw_page)
        (self.work/'report-content.txt').write_text('\n\n'.join(self.text_parts)+'\n')
        return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('final_submission'))
    parser.add_argument('--work',type=Path,default=Path('runs/final-package-validation-20261123'))
    args=parser.parse_args()
    output=args.output.resolve();work=args.work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    expected={'epsilon_ball.py','perlin_fog.py',REPORT}
    for stem in STEMS:
        for norm in NORMS:
            expected.update((f'{stem}_{norm}.png',f'raw_{stem}_{norm}.png'))
        expected.add(f'{stem}_perlin.json')
    assert len(expected)==48
    if output.exists() and ({p.name for p in output.iterdir()}-expected):
        raise ValueError('preserve existing output containing unexpected files')
    fonts()
    report=Report(output,work)
    pdf=report.build()
    output.mkdir(exist_ok=True)
    for name in expected:
        source=pdf if name==REPORT else ROOT/name if name.endswith('.py') else ROOT/'submission'/name
        shutil.copyfile(source,output/name)
    assert {p.name for p in output.iterdir()}==expected
    manifest={name:dict(bytes=(output/name).stat().st_size,sha256=sha(output/name)) for name in sorted(expected)}
    (work/'package-manifest.json').write_text(json.dumps(dict(directory=str(output),file_count=48,
        report_source=str(Path(__file__).relative_to(ROOT)),figures=report.figure_files,files=manifest),indent=2)+'\n')
    print(json.dumps(dict(directory=str(output),files=48,pdf_bytes=(output/REPORT).stat().st_size,
                         classification_figures=len(report.figure_files)),indent=2))


if __name__=='__main__':
    main()
