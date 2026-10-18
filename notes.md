# Project 01: The Honest Model

Notes that explain this project from zero. You do not need to know any machine learning
to read them. By the end, every file in this project should make sense to you.

---

## Start here

### What this project does, in one paragraph

A lender gives loans. Some people repay; some have trouble repaying. Before saying yes
to a new applicant, the lender would like a warning: *how likely is this person to have
trouble?* This project teaches a computer to give that warning by learning from 307,511
past applicants whose outcomes are already known. But the real goal is not a clever
model. The real goal is an **honest score**: a measurement of how good the model is that
will still be true when it meets new applicants it has never seen. Most of the mistakes
in machine learning are not about building the model; they are about *measuring* it
in a way that accidentally cheats.

### Map: which file is explained where

| File | What it does | Chapter |
|---|---|---|
| `src/honest_model/config.py`, `config.yaml` | Settings in one place, checked at start-up | 11 |
| `scripts/download_data.py` | Downloads the data (only if not already there) | 11 |
| `src/honest_model/memory.py` | Makes the table use less computer memory | 2 |
| `src/honest_model/features.py` | Summarises the extra tables into one row per applicant | 2 |
| `scripts/benchmark_vectorization.py` | Compares slow loops with fast whole-array maths | 3 |
| `scripts/train_naive.py` | The deliberately *cheating* version | 6 |
| `scripts/train.py` | The honest version | 4, 6 |
| `scripts/validate.py` | Tests the model five times in different ways | 8 |
| `scripts/hyperparameter_tuning.py` (v1) | First attempt at choosing settings fairly (has mistakes, kept for study) | 9 |
| `scripts/hyperparameter_tuning_v2.py` (v2) | The corrected version | 9 |
| `tests/` | Automatic checks that the code does what it should | throughout |

### How to read these notes

Read chapters 1 to 9 in order; each one builds on the previous. Every chapter has the
same shape: **the problem**, **an everyday picture**, **the idea**, **where it is in our
code**, and **what we found**. Words in **bold** the first time they appear are defined
right there, and again in the Glossary at the end. Each chapter ends with "Check
yourself" questions; click to reveal the answers.

---

## 1. What is machine learning?

### The problem

How can a computer decide something it was never given rules for? Nobody can write down
an exact rule like "people with income below X and age below Y will not repay". Real life
is too messy.

### An everyday picture

Think of a young doctor. Nobody gives them a rulebook for every patient. Instead, they
see thousands of past patients: their symptoms, and what turned out to be wrong. Over
time they *learn the patterns*. Then, when a new patient walks in, they make a good
guess.

### The idea

Machine learning does the same thing with data.

- Each past applicant is one **row** in a table.
- The facts we know about them (income, age, loan amount, and so on) are the
  **features**. They play the part of the symptoms.
- The outcome we want to predict is the **target**. Here it is the column `TARGET`: 1 if
  the person had trouble repaying, 0 if not. It plays the part of the diagnosis.
- **Training** means showing the computer many rows *with* their target, so it can find
  patterns that connect features to outcome.
- The result of training is a **model**: the learned pattern, which can now be given a
  new applicant's features and return a **prediction**, here a probability of trouble.

There is one catch, and this whole project is about it. A model can look brilliant on
the people it learned from, simply by memorising them. To know if it really learned,
we must test it on people it has **never seen**. So we hide some rows before training:
the **training set** is what the model learns from, and the **test set** is the exam it
takes afterwards.

### Where it is in our code

Our data is the file `data/raw/application_train.csv`:

- 307,511 rows, one per applicant.
- 122 columns. We currently use the 99 columns that hold numbers as features.
- 24,825 applicants had trouble repaying: **8.07%**. So the two outcomes are very
  unequal. Only about one person in twelve is a "1". This matters later (Chapter 8).

Every training script starts the same way: load the table, separate `TARGET` (the
answer) from the features, and drop `SK_ID_CURR`, which is just an ID number. An ID
number tells you nothing about a person, so the model should never see it.

### Check yourself

<details><summary>Why do we drop the ID column before training?</summary>

An ID is just a label, like a ticket number. It has no real connection to whether
someone repays. If the model saw it, it might find accidental patterns in it that would
never hold for new applicants.
</details>

<details><summary>Why can't we judge the model on the same rows it learned from?</summary>

Because it may have simply memorised them, like a student who memorised last year's
answers. The only fair test is on rows it has never seen.
</details>

---

## 2. The data: one row per person

### The problem

Our main table has one row per applicant. But other tables (for example `bureau.csv`,
the person's loans with other lenders) have **many** rows per applicant: one per past
loan. How do we use that information without breaking the main table?

### An everyday picture

Imagine a class register with one line per student, and a separate notebook listing
every test each student ever took. You cannot paste the whole notebook into the
register. You would write a **summary** next to each name instead: "number of tests
taken", "worst score", "average score".

### The idea

1. **Summarise first.** For each applicant, collapse their many rows into one row of
   summary numbers: how many past loans, how many still active, the largest debt, and
   so on. This is called **aggregation**.
2. **Then join.** Attach the summary to the main table using the applicant's ID. This
   is called a **merge**.

**The trap: row explosion.** If you merge *before* summarising, each applicant's single
row is copied once for every matching row in the other table. An applicant with twelve
past loans becomes twelve rows. The table silently grows from about 300,000 rows to
millions, with no error message at all.

**The second trap: information from the future.** Some rows in the other tables
describe events that happened *after* the loan application. The lender could not have
known them at decision time, so the model must not see them either. The rule we used:
keep only events dated strictly before the application day (for bureau loans,
`DAYS_CREDIT < 0`). The application day itself (`= 0`) is excluded, because an event on
that day might be the very loan being decided.

### Where it is in our code

- `src/honest_model/features.py` holds one function per extra table
  (`aggregate_bureau`, `aggregate_pos_cash`, `aggregate_installments`,
  `aggregate_credit_card`, `aggregate_previous_applications`). Each one keeps only past
  events, creates simple yes/no helper columns (such as "is this loan active?"),
  summarises everything to one row per applicant, and names the results with a prefix
  such as `BUREAU_`.
- **Status:** these summary features are built and tested, but the training scripts do
  **not** use them yet. Training so far uses only the main table.
- `src/honest_model/memory.py` makes tables smaller in memory, by storing numbers in
  smaller types and repeated text as categories. On the main table it cut memory from
  504.99 MB to 96.49 MB (80.89% smaller). That matters because later scripts train
  hundreds of models on this table.

### Watch out

- To check that a table really has one row per person before merging, count the rows
  per ID: `df.groupby("SK_ID_CURR").size().max()` must be 1.
- After merging, the main table's row count must not change.

### Check yourself

<details><summary>You merge a summary table and the row count stays the same. Does that prove the merge was right?</summary>

No. It proves no rows were duplicated. The summary could still contain information from
the future (events after the application) and be wrong in a much quieter way.
</details>

---

## 3. Making the computer fast

### The problem

Training means doing the same calculation millions of times. Done the slow way, a job
can take hours instead of seconds.

### An everyday picture

Paying for groceries one item at a time, with a separate payment for each, versus
putting everything in one basket and paying once.

### The idea

A **loop** in Python handles one number at a time, and Python does extra checking work
for every single one. **Vectorization** means handing the whole column of numbers to
NumPy in one go. NumPy does the work in fast compiled code, all at once.

**Broadcasting** is NumPy's way of combining arrays of different shapes without copying
data. For example, it can subtract each column's average from every row in one step.

### Where it is in our code

`scripts/benchmark_vectorization.py` times three real calculations both ways:

| Calculation | Loop | Vectorized | Faster by |
|---|---|---|---|
| Recency decay (100k loans × 3 windows) | 0.2810 s | 0.0110 s | 25.5× |
| Relative credit gap (100k credits × 5 tiers) | 0.3597 s | 0.0119 s | 30.3× |
| Standardising a matrix (100k rows × 10 features) | 1.2136 s | 0.0643 s | 18.9× |

### Watch out

Broadcasting saves memory on the *inputs*, but the *result* is a full new array. Always
work out the size of the result first; combining two long arrays can produce a result
far too big for memory.

---

## 4. Cleaning the data before training

### The problem

Many cells in the table are empty, and the columns use wildly different scales (ages in
tens, loan amounts in hundreds of thousands). How do we fill the gaps, and *what is
allowed to influence how we fill them?*

### An everyday picture

A teacher marking a test finds some answers left blank and decides to give those the
"typical" mark. The important question is: typical *of whom*? If the teacher works out
"typical" using next year's class as well, that is using information they should not
have.

### The idea

- **Imputation** means filling empty cells. We fill each column's gaps with its
  **median**, the middle value when all values are sorted. We chose the median over the
  average (the mean) because money columns contain a few enormous values that drag the
  average far away from a normal applicant. The median ignores those extremes.
- **Scaling** means putting columns on a similar scale, so that "large" means the same
  thing in every column.

Each of these tools works in two steps:

1. **fit**: *learn* the rule from data (for example, "the median income is X").
2. **transform**: *apply* the rule (fill the blanks with X).

The golden rule: **learn the rule from the training set only, then apply that same rule
to everything else.** The test set represents the future. In real life the future has
not happened yet, so it cannot have helped set the rule. And in production, applicants
arrive one at a time; you cannot calculate a median from a single person, so you must
use the one stored from training.

A **pipeline** packs the cleaning steps and the model into one sealed box. Whenever you
train the box on some rows, every step inside learns its rule from those rows only. That
makes the golden rule automatic instead of something you have to remember.

### Where it is in our code

`scripts/train.py`:

1. Splits the data into 80% training rows and 20% test rows **first**.
2. Builds a `Pipeline`: median imputer → scaler → model.
3. Trains the pipeline on the training rows only, so the median and scale are learned
   from them alone.
4. Predicts on the test rows and scores the result.

### Check yourself

<details><summary>Why must the median come from the training rows only?</summary>

Because the test rows stand in for future applicants. If they helped set the median, a
little information about the "future" has leaked into training, and the test is no
longer a fair exam.
</details>

---

## 5. How do we score a model?

### The problem

The model outputs a probability of trouble for each applicant. How do we turn thousands
of predictions into one number that says "how good"?

### An everyday picture

Pick two applicants at random: one who really had trouble repaying, and one who did not.
Ask the model which of the two is riskier. A good model points to the right person most
of the time.

### The idea

That is exactly what the **ROC-AUC** score measures: across all such pairs, how often
the model ranks the person who had trouble as riskier than the person who did not.

- **0.5** means a coin flip: the model has learned nothing.
- **1.0** means it always ranks them correctly.
- Our models score about **0.75**: right about three times out of four.

We use ROC-AUC instead of plain "percentage correct" because only 8.07% of applicants
had trouble. A lazy model that always says "no trouble" would be correct 91.93% of the
time while being completely useless. ROC-AUC is not fooled by that.

### Where it is in our code

Every script scores with `roc_auc_score` (in `train.py`) or `scoring="roc_auc"` (in the
cross-validation scripts).

---

## 6. Accidental cheating: data leakage

### The problem

What if the model's score is high only because it secretly saw part of the exam?

### An everyday picture

A student scores brilliantly in a mock exam, but it turns out they had seen the question
paper the night before. The score tells you nothing about how they will do in the real
exam.

### The idea

**Data leakage** is when information the model would not have in real life sneaks into
training or into the measurement. The score then looks better than the model will
really perform. There are four common ways it happens:

| Kind of leak | What happens | Our example |
|---|---|---|
| Peeking at the test set while cleaning | The median or scale is learned from all rows, test rows included | `train_naive.py` |
| Using information from the future | A feature describes events after the decision | bureau events on or after the application day (Chapter 2) |
| A badly chosen split | The same person, or later periods, end up on both sides of the split | Chapter 8 |
| Choosing settings using the test | Picking the best settings by looking at the same scores you report | Chapter 9 |

Leaks from the future are usually the most damaging, because they carry clues about the
outcome itself.

### Where it is in our code

We built the project both ways on purpose:

- `scripts/train_naive.py` (Day 6) **cheats**: it fills blanks and scales using all
  307,511 rows, and only then splits into training and test.
- `scripts/train.py` (Day 7) is **honest**: it splits first and uses a pipeline.

### What we found

| Version | ROC-AUC |
|---|---|
| Cheating (`train_naive.py`) | 0.75061 |
| Honest (`train.py`) | 0.75086 |
| Difference | +0.00025 (the honest one is slightly *higher*) |

So on this data, this particular leak made almost no difference. That is because, with
about 300,000 rows, adding the test rows barely moves a median, and the difference is
smaller than ordinary random variation (Chapter 8). The leak is still a real mistake in
principle. It matters a lot with small datasets, and with cleaning steps that look at
the target itself.

### Watch out

A pipeline protects you from the first kind of leak only. It has no idea *how* you split
the data or *where* your features came from.

### Check yourself

<details><summary>Name a leak that a correctly built pipeline cannot catch.</summary>

A badly chosen split. For example, if loans are ordered in time and you split randomly,
the model trains on recent loans and is tested on older ones. In real life you always
predict the future from the past. The pipeline cleans correctly, but it is cleaning a
split that is already unfair.
</details>

---

## 7. The model: decision trees and LightGBM

### The problem

What kind of "pattern learner" should we use for a table of numbers about people?

### An everyday picture

The game *Twenty Questions*. You find the answer with a chain of yes/no questions: "Is
it alive? Is it bigger than a cat?" A **decision tree** does the same: "Is income below
some amount? Then, has this person missed a payment before?" Each answer narrows the
group down until it can say "risky" or "safe".

### The idea

- One tree on its own is easy to fool. **Gradient boosting** builds a whole team of
  small trees, one after another. Each new tree concentrates on the mistakes the
  previous trees made. Together they are far stronger than any single tree. (Day 10 builds
  this from scratch.)
- **LightGBM** is a fast, popular implementation of gradient boosting. We use its
  `LGBMClassifier`. It is quick because it groups each column's values into a limited
  number of bins instead of checking every possible value.
- Why trees suit this problem: risk is rarely a straight line. Having *some* borrowing
  history can be safer than having none at all *or* a great deal. Trees capture "sweet
  spots" like that naturally; straight-line models cannot.

Two useful facts about trees:

1. **Scaling makes no difference to a tree.** A tree only asks "is this value above or
   below a cut-off?" Rescaling a column changes the numbers but not their order, so the
   tree can ask exactly the same questions. Our scaler is therefore doing nothing useful
   for LightGBM. It only costs time.
2. **LightGBM can handle empty cells itself.** At each question it learns which way the
   "blank" cases should go. Filling blanks with the median throws away a possibly useful
   signal: sometimes *leaving a field blank* says something about the applicant.

**Open decision:** whether to keep the imputer and the scaler in front of LightGBM.
This is still to be decided and tested.

### Where it is in our code

`LGBMClassifier(random_state=42)` is the last step of every pipeline.

---

## 8. One exam is not enough: cross-validation

### The problem

A single 80/20 split gives one score. But maybe that particular 20% happened to be easy,
or hard. How sure can we be about that one number?

### An everyday picture

You would not judge a student on one test. You would give them several tests and look
at the average, and at how much the results vary.

### The idea

**Cross-validation** cuts the data into 5 equal parts, called **folds**. Then it runs 5
rounds. In each round, a different fold is the test and the other 4 are for training.
That gives 5 scores: their **average** is our estimate, and their **spread** tells us how
much the result jumps around by luck.

How the 5 folds are made is a real decision:

| Way to split | How | Use it when |
|---|---|---|
| `KFold` (shuffled) | Shuffle all rows, then deal them into 5 folds | Rows are independent, one per person |
| `StratifiedKFold` | Like KFold, but each fold gets exactly the same share of "1"s | The "1"s are rare and the data is small |
| `GroupKFold` | All rows of one person stay in the same fold | One person has several rows |
| `TimeSeriesSplit` | Train on earlier rows, test on the next block | Rows are ordered by real dates |

**Do we need the stratified version?** By luck alone, the share of "1"s in a random fold
moves by about √(p × (1 − p) / n), where p is the share of "1"s and n is the fold size.
For us: √(0.0807 × 0.9193 / 61,502) ≈ 0.0011. So every random fold already contains about
8.07% ± 0.11 percentage points of defaulters. Each fold already looks like the whole
dataset, so stratifying has nothing to fix. With a small dataset or a very rare outcome,
it would matter.

### Where it is in our code

`scripts/validate.py` (Day 8) tries three of these ways on the same pipeline:

| Way | Average ROC-AUC | Spread (std) |
|---|---|---|
| `KFold`, shuffled | 0.75037 | 0.00144 |
| `GroupKFold` by applicant ID | 0.75037 | 0.00144 |
| `TimeSeriesSplit` | 0.74363 | 0.00680 |

What this means:

- **KFold and GroupKFold are identical** because every applicant appears exactly once
  (307,511 rows, 307,511 different IDs). Grouping by a unique ID changes nothing.
- **TimeSeriesSplit is not valid here.** It assumes the rows are in date order, but this
  table has no real dates. The `DAYS_...` columns count days *relative to each person's own
  application*, so they cannot put two different people in time order. The lower score
  mostly comes from something else: its training sets grow from about 51,000 rows in the
  first round to about 256,000 in the last. Fold scores rose from 0.73391 to 0.75182 as
  the training data grew. The rounds are not comparable, which also explains the bigger
  spread.

**Decision:** plain shuffled `KFold`, 5 folds.

### Watch out: a silent default

If you write `cv=3` (just a number) instead of a splitter, scikit-learn quietly decides
for you. For a classification model it uses `StratifiedKFold` **without shuffling**. That
is exactly the mistake in v1 (Chapter 9). Always pass the splitter itself, for example
`KFold(n_splits=5, shuffle=True, random_state=42)`, so anyone reading the code can see
the choice.

### Check yourself

<details><summary>Someone says "TimeSeriesSplit scored lower, so the model gets worse over time." What's wrong with that?</summary>

Two things. First, this table has no real dates, so the row order is not time order.
Second, TimeSeriesSplit trains on much less data in its early rounds, and less training
data alone lowers the score.
</details>

---

## 9. Choosing settings without fooling yourself

This is the longest chapter, because it is where our biggest mistake happened.

### The problem

A model has settings you choose before training: how many trees, how deep each one may
grow, how big a step each tree takes. These are called **hyperparameters**. We tried 6
combinations and kept the best one. But here is the question: **is the best one's score
an honest measure of how good it is?**

### An everyday picture

Six archers each shoot one round, and you crown the top scorer. Some of the winner's
score is skill, but some is luck: a lucky gust of wind. If you have the winner shoot
again, they will usually score a little lower. The winning score was **optimistic**,
because you chose it *for being the highest*.

### The idea

**Single CV** (the normal way): score all 6 combinations with cross-validation, pick the
best, and report its score. That reported score is optimistic, for the archer reason.
This bias is called **selection bias**. It grows when you try more combinations, and when
the scores are noisier.

**Nested CV** (the honest way) adds an outer loop, so the choosing and the final
measuring never use the same data:

```
For each of 5 outer rounds:
  1. Lock away 20% of the data as a "vault".
  2. On the other 80%: run a complete search (cross-validation over all 6
     combinations) and pick a winner for this round.
  3. Retrain that winner on the full 80%.
  4. Score it on the vault, which played no part in choosing.
Report: the average of the 5 vault scores.
```

The difference between the two reported scores is the **optimism gap**.

| | Single CV | Nested CV |
|---|---|---|
| What chooses the settings | the 5 folds | an inner search inside each outer round |
| What measures the result | the **same** 5 folds | a vault that took no part in choosing |
| The score means | the best of 6, so a little lucky | how well "search, then pick the winner" really works |

Two more points:

- **Different rounds may pick different winners.** That is normal. Nested CV does not
  measure one particular setting. It measures the *whole procedure*: "run this search and
  use whatever wins".
- **For the final model**, you run the search once on *all* the data, train the winner on
  all the data, and use that. The nested score is what you *report* as its expected
  performance.

### Where it is in our code: two versions, kept side by side on purpose

**Version 1: `scripts/hyperparameter_tuning.py` (kept unchanged for study)**

It reported single CV **0.74970**, nested CV **0.74931**, a gap of **+0.00039**. We first
wrote "single CV overestimates". That conclusion did **not** hold up, for three reasons:

1. **The gap was smaller than ordinary luck.** In Chapter 8 the fold scores varied by
   0.00144, about four times the size of the gap.
2. **It was one run.** One run gives one number, and one number cannot tell you how much
   it would change with a different shuffle. (Running v1 again gave exactly the same
   numbers twice, because with every seed fixed it repeats the same shuffle. See
   "seeds" below.)
3. **The two procedures were not set up the same way.** v1 wrote `cv=3` for the inner
   search, which silently became an unshuffled stratified 3-fold split, while the other
   loops used shuffled 5-fold KFold. So the gap could have come from that difference
   instead of from nesting.

It was also expected to be small. Six combinations is a short list, and some of them
are nearly the same model (in LightGBM, a depth limit of 5 or 10 makes little difference
when each tree is already capped at 31 leaves). Few real choices means little luck to
exploit.

**Version 2: `scripts/hyperparameter_tuning_v2.py` (the corrected version)**

1. All three loops share **one** splitter: `KFold(5, shuffle=True, random_state=seed)`.
   Now the two procedures differ only in being nested or not.
2. The model's own seed stays fixed at 42, so the only thing that changes between runs
   is how the rows are shuffled.
3. The whole experiment is repeated with 10 different seeds, which gives 10 gaps instead
   of 1.
4. Those 10 gaps are summarised and tested, as explained next.

### Seeds: what "seed 0, 1, 2…" means

A computer cannot truly shuffle. It produces numbers that only *look* random, starting
from one starting number: the **seed**. The same seed always gives the same shuffle.
A different seed gives a different shuffle.

Picture the 307,511 applicants as a deck of cards. `KFold(shuffle=True,
random_state=seed)` shuffles the deck with that seed and deals it into 5 piles: the 5
folds. Seed 0 is one shuffle, seed 1 another, and so on. The data never changes; only
*who lands in which fold* changes. If the gap is real, it should appear whatever the
shuffle. If it is luck, it will jump around, and may even turn negative.

### Is the gap real? Standard deviation, standard error, and the t-distribution

**An everyday picture:** measuring your height with a slightly unreliable tape measure.

- How far off **one** measurement usually is: that is the **standard deviation**. It is a
  property of the tape, and measuring more often does not improve the tape.
- Measure several times and take the **average**. The errors partly cancel out, so the
  average is closer to the truth. How far off the *average* usually is: that is the
  **standard error (SE)** = standard deviation ÷ √(number of measurements). It shrinks as
  you measure more. Four times as many measurements halves it.

Our "measurement" is one run, and the thing measured is the gap. We want to know about
the *average* gap, so we use the SE.

**The test:** compute **t = average gap ÷ SE**. If t is bigger than a threshold, the gap
is unlikely to be luck.

**What threshold?** With many runs, about 2. With few runs, the standard deviation itself
is only a rough guess (two numbers that happen to be close together look "very
consistent" by luck). So the threshold must be higher. The **t-distribution** is the bell
curve corrected for this: it expects surprises more often when there are few runs.

| Number of runs | 2 | 3 | 5 | 10 | 30 | very many |
|---|---|---|---|---|---|---|
| Threshold (95% sure) | 12.71 | 4.30 | 2.78 | **2.26** | 2.05 | 1.96 |

Two rules for wording the result:

- If it passes: "single CV is optimistic on this dataset".
- If it fails: "no gap was detected", **not** "there is no gap". Not finding something
  does not prove it is absent.

Also: decide the number of runs *before* starting, and judge only at the end. Stopping
early as soon as the result "looks real" is selection bias all over again.

**Limit:** every run uses the same 307,511 people; only the shuffle changes. So this
tests luck in the split, on this dataset. It says nothing about a different group of
applicants.

### What we found (v2)

10 seeds were planned. The run was stopped after **8** when the computer session ended. That
had nothing to do with the results, so the 8 completed runs are not cherry-picked and are
used as they are. Seed by seed:

| Seed | Single CV | Nested CV | Gap |
|---|---|---|---|
| 0 | 0.75001 | 0.74945 | +0.00055 |
| 1 | 0.75012 | 0.74973 | +0.00038 |
| 2 | 0.75011 | 0.74977 | +0.00035 |
| 3 | 0.74986 | 0.74959 | +0.00027 |
| 4 | 0.75001 | 0.74995 | +0.00006 |
| 5 | 0.74990 | 0.74992 | −0.00002 |
| 6 | 0.75031 | 0.74981 | +0.00050 |
| 7 | 0.74977 | 0.74963 | +0.00014 |

| Summary (8 runs) | Value | How it was worked out |
|---|---|---|
| Average gap | **+0.00028** | sum of the 8 gaps ÷ 8 |
| Standard deviation | 0.00021 | spread of the 8 gaps (divided by 8 − 1) |
| Standard error | 0.00007 | 0.00021 ÷ √8 |
| t | **3.85** | average ÷ standard error, using unrounded values (rounded ones give about 4) |
| Threshold for 8 runs | 2.36 | t-distribution, 95% |

**Verdict: the gap is real.** 3.85 is well above 2.36, and 7 of the 8 gaps are positive. On
this dataset, choosing the best of 6 settings with single CV overstates ROC-AUC by about
**0.0003** on average. That is real, but tiny, as expected from a short list of
near-duplicate settings. v1's +0.00039 pointed the right way, but one run could not have
shown that.

What a real gap means in practice: the model is not broken. The single-CV score is just a
little too high, so the honest number to report is the nested one. And any tuning
"improvement" smaller than about 0.0003 on this data could just be luck.

*Note on the script:* v2 prints its own verdict using a fixed bar of 2. The verdict above
uses the correct t-distribution bar for 8 runs (2.36), as explained earlier in this
chapter. Both say "real" here.

### Check yourself

<details><summary>Why can't one run tell you whether a gap is real?</summary>

One run is one shuffle, so one random draw. Running it again with the same seed repeats
the same shuffle and gives the same number. You need many different shuffles to see how
much the gap jumps around.
</details>

<details><summary>Does using 5 inner folds instead of 3 make the final nested score less variable?</summary>

No. The final score is the average of the 5 *outer* scores, and the inner folds don't
change that. More inner folds make the *choice* of winner in each round more reliable,
and give the inner models more data to train on (64% of all rows instead of 53%). It
costs more training runs: 155 instead of 95 per experiment.
</details>

---

## 10. Getting the same result every time

### The problem

You run the same code twice and get slightly different numbers. Which one is right?

### The idea

Many steps involve randomness: shuffling rows, the order columns come in, some model
options. A **seed** fixes each one. But a seed only fixes the one thing it is attached to:

| Seed | Controls |
|---|---|
| The splitter's seed (`KFold(random_state=…)`) | which rows go into which fold |
| The model's seed (`LGBMClassifier(random_state=…)`) | randomness inside training |

So the right question is not "is my model reproducible?" but "is my **whole pipeline**
reproducible?"

If numbers change between identical runs, check in this order:

1. Are the splits the same?
2. Are the feature tables the same?
3. Are all the seeds fixed?

Running on several CPU cores can also cause tiny differences in the last decimal places.

---

## 11. The project toolbox

Short notes on the tools around the code:

- **`pyproject.toml`** lists the libraries the project needs. **`uv.lock`** records the
  exact version of every library, including the ones those libraries depend on, so the
  project installs identically on any computer.
- **One environment per computer system.** A `.venv` created on Windows cannot be used
  from WSL (Linux), and the other way round. Running `uv` from the wrong side tries to
  rebuild it, and can break it halfway. Pick one side and stay there.
- **`config.yaml` + `config.py`**: settings live in one file and are checked by Pydantic
  as soon as the program starts. A typo fails in the first second, not hours later.
- **`download_data.py`** skips the download if the files are already there.
- **Pre-commit hooks** check formatting and stop big data files before they enter Git.
- **`.gitignore`** keeps data, models, `.venv/` and secrets out of Git.
- **`uv run python scripts/...`** runs a script inside the project's environment, so the
  project's own package (`honest_model`) can be imported.

---

## Mistakes and lessons

| # | What happened | Why (the thinking error) | Lesson | Status |
|---|---|---|---|---|
| 1 | `uv init --package <name>` made a nested folder | Thought the argument was the package name; it is a folder | Use `uv init --package --name <name> .` | Fixed |
| 2 | `import pyyaml` failed | Assumed the install name equals the import name | Install `pyyaml`, import `yaml` | Fixed |
| 3 | Overrode Pydantic's `model_validate` and broke it | Thought nested settings needed manual parsing | Do not override framework methods without knowing what they already do | Fixed |
| 4 | Put an expression inside a pandas named aggregation | Named aggregation needs a *column name*, not an expression | Make a helper column first, then aggregate it | Fixed |
| 5 | Tests checked column names that no longer exist after aggregation | Mixed up the input columns and the output columns | Test the output names | Fixed |
| 6 | Chose mean imputation without a reason | Took a default without asking what it assumes | Median for skewed money data (Chapter 4) | Fixed |
| 7 | Wrote a script that predicted but never scored | Stopped at the predictions | Always: score → compare → report the difference | Fixed |
| 8 | Wrote `cv=3` in v1's inner search | Did not know a plain number becomes an unshuffled stratified split | Always pass the splitter object (Chapter 8) | Fixed in v2 |
| 9 | Compared two procedures that differed in more than one way | Changed several things at once | Change exactly one thing per comparison | Fixed in v2 |
| 10 | Called a gap of +0.00039 "optimistic" | Read the sign of one number without checking it against luck | Compare every difference with its noise (Chapter 9) | Resolved: v2 over 8 seeds shows the gap is real but tiny (+0.00028, t = 3.85) |
| 11 | The grid had near-duplicate settings | Did not know how `max_depth` and `num_leaves` interact | Check that candidates are really different | Open |
| 12 | Notes quoted leak sizes that were never measured | Wrote down intuition as if it were fact | Only measured numbers go in the notes | Fixed |
| 13 | Imputer and scaler sit in front of LightGBM | Copied a general recipe without asking whether trees need it | Ask whether each step is needed for *this* model (Chapter 7) | Open |
| 14 | While fixing v1, changed the model's seed instead of the inner splitter, and used `seed` without defining it | Did not separate the two kinds of seed | Know what each `random_state` controls (Chapter 10) | Fixed in v2 |
| 15 | Ran `uv` from WSL on a Windows-made `.venv`, which half-deleted it | Did not know an environment belongs to the system that created it | One side only (Chapter 11) | Fixed (environment rebuilt) |
| 16 | `aggregate_pos_cash` divides the failed-payment count **by itself** to get `POS_PAYMENT_FAILURE_RATE`, so the result is always 100 or blank | Probably copied the line and forgot to change the divisor | Check every new feature's values: a rate that is always 100 is a bug | **Open** (found while rewriting these notes) |

---

## Rules of thumb

**Always**
- Split first, then fit every cleaning step on the training rows only, using a pipeline.
- Build features only from information that existed before the decision.
- Pass the splitter as an object. Never pass a bare number to `cv=`.
- Change one thing at a time when comparing.
- Compare every difference with its noise before calling it a finding.
- Report a nested (or held-out) score when you have tuned settings.
- Train the final model on all the data with the chosen settings.
- Lock library versions, fix every seed, and validate settings at start-up.

**Never**
- Never fit a cleaning step on data that includes the test set.
- Never report the best score from a search as the expected real-life performance.
- Never pick "the best fold", and never ship a model from inside cross-validation.
- Never use `TimeSeriesSplit` on rows that are not in real date order.
- Never write down a number you did not measure.
- Never commit data, models, `.env` or virtual environments.

---

## Results log

| Day | Script | What was measured | Result | Meaning |
|---|---|---|---|---|
| — | `memory.py` (logged by the tuning scripts) | Memory of the main table | 504.99 MB → 96.49 MB (80.89% smaller) | Hundreds of training runs fit comfortably |
| 5 | `benchmark_vectorization.py` | Loops vs vectorized | 18.9× to 30.3× faster | Always vectorize |
| 6 | `train_naive.py` | ROC-AUC with the cleaning leak | 0.75061 | Reference "cheating" score |
| 7 | `train.py` | ROC-AUC, honest | 0.75086 (+0.00025) | The leak was negligible on this data |
| 8 | `validate.py` | KFold, 5 folds | 0.75037 ± 0.00144 | Our noise level |
| 8 | `validate.py` | GroupKFold | 0.75037 ± 0.00144 | Same, because IDs are unique |
| 8 | `validate.py` | TimeSeriesSplit | 0.74363 ± 0.00680 | Not valid: no real dates |
| 9 | `hyperparameter_tuning.py` (v1) | Single vs nested, one seed | 0.74970 vs 0.74931, gap +0.00039 (reproduced exactly twice) | One run cannot prove it; see v2 |
| 9 | `hyperparameter_tuning_v2.py` | Gap over 8 seeds (10 planned; stopped by session end) | mean +0.00028, SE 0.00007, t = 3.85 > 2.36 | Single CV is optimistic, but only by about 0.0003 |

---

## Questions and answers

**What is nested CV protecting me from?** *(Day 9 Prove-it)*
From **selection bias**: reporting a score that is too high because the same data both
*chose* the winner and *measured* it.
1. In single CV, the same 5 folds do two jobs: they pick the best of the 6 settings, and
   they give its score.
2. Every score contains some luck. Picking the highest tends to pick a setting that was
   good **and** lucky on those particular folds.
3. So the winner's score is too high. The folds "favour" it, because it was chosen on them.
4. Nested CV gives the two jobs to different data. Inside each outer round, the inner
   loop chooses using only the 80%. Then the 20% vault, which took no part in choosing,
   measures the winner. The vault cannot be lucky in the winner's favour.

Our measurement: on this data, the bias was real but tiny (about 0.0003; Chapter 9).

**When is the extra compute of nested CV not worth it?** *(Day 9 Prove-it)*
It has to be decided **before** paying for it, so the gap itself cannot be the test: you
only know the gap after running nested CV. Instead, look at what makes the bias large.

| Nested CV is less worth it when… | Nested CV is worth it when… |
|---|---|
| you try **few, clearly different** settings | you try **many** settings (hundreds) |
| the data is **large**, so each score is stable | the data is **small**, so each score is noisy |
| you only need to **pick** a model | you must **report** an honest number that someone relies on |
| you can afford to lock away a separate **test set** at the start | the data is too small to spare a locked test set |

The cheaper alternative, when data is plentiful: lock away a test set **before any
tuning**, tune on the rest, and score the final winner on the locked set **once**.

For this project: 307,511 rows, 6 near-duplicate settings, 155 training runs per experiment
instead of 31, and a correction of only about 0.0003. Here, nested CV mostly confirmed that
single CV was nearly honest. It becomes clearly worth it with a large search on a small
dataset, or whenever the reported number is a promise to someone.

**Why does putting the imputer inside the pipeline change the score?** *(Day 7)*
In `train_naive.py`, the median was calculated from all 307,511 rows, including the
61,503 that later became the test set. Those test rows helped fill the training data.
In `train.py`, the pipeline only learns the median from the 246,008 training rows, so the
test rows are a truly unseen exam. On this data the effect was tiny (Chapter 6).

**What leak does a correct pipeline still miss?** *(Day 7)*
A badly chosen split. A pipeline controls *where* each step learns, but not *how* the
rows were divided. Random splits on time-ordered data let the model learn from the
future.

**Plain KFold or stratified?** Plain. Random folds already hold 8.07% ± 0.11 percentage
points of defaulters, so there is nothing to fix (Chapter 8).

**Why 5 inner folds instead of 3?** It picks the winner more reliably and trains inner
models on more data. It does *not* make the final score less variable (Chapter 9).

**How do I know if a gap is real?** Repeat with many seeds, average the gaps, compute
t = average ÷ SE, and compare with the t-distribution threshold for that many runs
(Chapter 9).

**What is the SE?** How far the *average* of several measurements is likely to be from
the truth. It equals the standard deviation ÷ √(number of measurements) (Chapter 9).

**What is a seed?** The starting number for a shuffle. Same seed, same shuffle (Chapter 9).

**What is the t-distribution?** The bell curve corrected for having only a few
measurements. It sets a higher bar when there are few runs (Chapter 9).

**If the optimism gap turns out to be real, is that good or bad?** Neither. It is
*information about your measurement*, not a fault in the model. A real gap means that
the score from single CV is a little too high, so you report the nested score instead.
What matters is the **size**:
- **Small gap** (ours is about 0.0003): the search did not exploit luck much, so single CV
  was nearly trustworthy for this grid. Nothing needs fixing; just report the honest number.
- **Large gap:** a warning that the search is "winning by luck". That happens with too many
  candidates, too little data, or a noisy score. Fixes: try fewer and more different
  candidates, use more data or folds, and always report the nested (or held-out) score.

There is also a practical rule: if two models differ by **less than the gap**, the
"better" one may only have been luckier. Don't trust a tuning win smaller than the gap.

**Earlier questions (Days 3–5), in one line each:**
- A merge multiplies rows when the right-hand table has several rows per key. Check with
  `groupby(key).size().max()`.
- `.apply()` on groups is slow because it runs Python code once per group, instead of
  fast built-in code.
- The shape of `A[:, None, :] * B[None, :, :]` for A (4, 3) and B (5, 3) is (4, 5, 3).

### Concepts to revise

- Turning a one-off experiment into a repeated one (v2 was written for me). Practise:
  load the data once, put one experiment in a function of `seed`, loop over seeds,
  summarise with the average, the standard deviation (`ddof=1`) and the SE.
- Testing whether a gap is real: standard deviation vs SE, the t-distribution.
- What inner folds control versus outer folds.
- scikit-learn's silent default for `cv=<number>`.
- Mean vs median imputation.
- Completing the evaluation: score → compare → report.
- Prove-it, Day 9: *how* nested CV prevents selection bias (the same folds choosing and
  measuring), and deciding *before* running whether nested CV is worth its cost.

---

## Glossary

| Word | Plain meaning |
|---|---|
| Aggregation | Summarising many rows into one (count, sum, max, average) |
| Broadcasting | NumPy combining arrays of different shapes without copying |
| Cross-validation | Testing a model several times, each time on a different part of the data |
| Decision tree | A model that decides through a chain of yes/no questions |
| Feature | A fact about each row that the model can use |
| Fit | Learning a rule from data |
| Fold | One of the equal parts the data is cut into for cross-validation |
| Gradient boosting | A team of small trees, each fixing the previous ones' mistakes |
| GroupKFold | Cross-validation that keeps all rows of one person in the same fold |
| Hyperparameter | A setting chosen before training, such as tree depth |
| Imputation | Filling empty cells |
| KFold | Cross-validation that shuffles the rows and deals them into k folds |
| Leakage | Information the model should not have, sneaking into training or measurement |
| LightGBM | A fast gradient-boosting library |
| Lock file (`uv.lock`) | Exact versions of every library, for identical installs |
| Median | The middle value when all values are sorted |
| Merge | Joining two tables on a shared ID |
| Model | The learned pattern that turns features into a prediction |
| Nested CV | Cross-validation with a search inside each round, so choosing and measuring never share data |
| Optimism gap | How much the single-CV score exceeds the nested score |
| Pipeline | Cleaning steps and model packed into one box, so each fit uses only its own training rows |
| ROC-AUC | How often the model ranks a real defaulter as riskier than a non-defaulter (0.5 = coin flip) |
| Scaling | Putting columns on a similar scale |
| Seed | The starting number for a computer's shuffle; same seed, same shuffle |
| Selection bias | The best of several noisy scores looks better than it really is |
| Standard deviation | How much one measurement typically varies |
| Standard error (SE) | How much the *average* of several measurements typically varies |
| StratifiedKFold | KFold where each fold gets the same share of each outcome |
| t-distribution | The bell curve corrected for few measurements; sets a higher bar |
| Target | The column we want to predict (`TARGET`) |
| Test set | Rows hidden from training, used as the exam |
| TimeSeriesSplit | Cross-validation that trains on earlier rows and tests on later ones |
| Training set | Rows the model learns from |
| Transform | Applying a learned rule to data |
| Vectorization | Doing maths on a whole array at once instead of one value at a time |

---

## Project recipe

A reusable plan for building a similar project (any "predict a yes/no outcome from a
table about people" problem) on new data.

**R1. When to use it.** Predicting a yes/no outcome from tabular data about people or
accounts (loan default, customer leaving, fraud), when the model will judge *future*
cases. Not for images, text or audio, nor for forecasting a value over time.

**R2. What makes it hard.** Almost every step can leak. Real differences are small, so
luck can pass for a finding. Library defaults make choices for you without telling you.

**R3. What you need.** A main table (one row per person, with the outcome), optional
extra tables (many rows per person, with dates), a data dictionary. Python with `uv`,
pandas, NumPy, scikit-learn and LightGBM. Chapters 1–10 of these notes.

**R4. Done means.** One pipeline trained on all the data; an honest (nested) score with
its spread; and evidence that it is honest: no leaks, and every claimed difference
bigger than its noise.

**R5. Steps.**

1. **Set up** so any run can be repeated: lock file, settings checked at start-up, data
   kept out of Git. *Check:* a fresh install works with one command.
2. **Build features** from the extra tables: past events only, summarised to one row per
   person, then merged. *Check:* one row per ID before the merge, and the same row count
   after.
3. **Build a deliberately cheating version** as a reference score.
4. **Build the honest version:** split first, then a pipeline. *Check:* the difference
   from step 3 is recorded and compared with noise.
5. **Choose how to split.** Real dates → time split. Repeated people → group split.
   Rare outcome on small data → stratified. Otherwise shuffled KFold. *Check:* the
   splitter is visible in the code, and the fold spread is recorded as the noise level.
6. **Tune honestly:** nested CV under the same splitter, repeated over several seeds,
   judged with the t-test. *Check:* the procedures differ in one thing only.
7. *(Later days: boosting from scratch, calibration, thresholds, interpretation,
   serving.)*

**R6. Decisions so far.**

| Decision | Choice | Why | Would change if |
|---|---|---|---|
| Time cut-off for extra tables | strictly before the application day | an event on that day may be the decision itself | dates are precise enough to order events within a day |
| Imputation | median | money columns have extreme values | values are roughly symmetric |
| Splitter | shuffled KFold, 5 folds | IDs are unique; no real dates; folds are already balanced | repeated people, real dates, or small data |
| Inner folds | 5 | more reliable choice; affordable for 6 combinations | many combinations or slow training |
| Imputer and scaler before LightGBM | open | — | — |
| Settings grid | open | — | — |

**R7. Final checklist.** Locked environment · settings checked at start-up · one row per
ID before every merge · only past events in features · no cleaning fitted on test rows ·
splitter passed explicitly everywhere · one change per comparison · noise level recorded ·
every claim bigger than its noise · nested score reported · final model trained on all
data.

**R8. On a new dataset.** *Keep:* the order of steps, pipelines, explicit splitters,
noise checks, nested scores. *Decide again:* the split, the date cut-off, whether the
model needs cleaning steps, the settings grid. *Measure again:* the outcome rate, rows per
ID in each table, the fold spread, the time per training run.

**R9. Looking back.** *(To be written at the end of the project.)*
