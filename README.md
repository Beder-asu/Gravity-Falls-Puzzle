# Jigsaw Puzzle Solver using Classical Computer Vision

## Project Overview

This project implements an **automatic jigsaw puzzle solver** using **only classical image processing techniques** — no AI/ML models are used. The solver takes a shuffled puzzle image and reconstructs the original image by analyzing piece boundaries and finding optimal piece arrangements.

### Key Results

| Puzzle Size | Perfect (100%) | Mean Accuracy | Time/Puzzle |
|-------------|----------------|---------------|-------------|
| 2×2         | 95.5%          | 95.5%         | < 0.1s      |
| 4×4         | 91.6%          | 91.6%         | ~0.3s       |
| 8×8         | 72.2%          | 72.2%         | ~3.0s       |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        JIGSAW PUZZLE SOLVER PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  1. INPUT    │ ──▶ │ 2. SEGMENT   │ ──▶ │ 3. ANALYZE   │ ──▶ │ 4. MATCH     │
    │  Shuffled    │     │  Extract     │     │  Compute     │     │  Best        │
    │  Puzzle      │     │  Pieces      │     │  Boundaries  │     │  Buddies     │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                          │
                                                                          ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  8. OUTPUT   │ ◀── │ 7. OPTIMIZE  │ ◀── │ 6. EXPAND    │ ◀── │ 5. GROW      │
    │  Solved      │     │  Border      │     │  Toroidal    │     │  A* Region   │
    │  Image       │     │  Scoring     │     │  Shifts      │     │  Growing     │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## Detailed Pipeline Description

### Step 1: Image Segmentation

**Function:** `segment_image(image, grid_size)`

The input shuffled puzzle image is divided into equal-sized pieces based on the grid size (2×2, 4×4, or 8×8).

```python
# Divide image into grid_size × grid_size pieces
pieces = []
for i in range(grid_size):
    for j in range(grid_size):
        pieces.append(image[i*piece_h:(i+1)*piece_h, j*piece_w:(j+1)*piece_w])
```

**Justification:** This step assumes pieces are arranged in a regular grid, which is the standard format for type-1 jigsaw puzzles (square pieces without interlocking edges).

---

### Step 2: Edge Feature Extraction

**Function:** `compute_dissimilarity_matrices(pieces, use_gradient=True)`

For each piece, we extract boundary features from all four edges:

#### 2.1 LAB Color Space Conversion
```python
lab = cv2.cvtColor(piece, cv2.COLOR_RGB2LAB)
```

**Why LAB?** The LAB color space is **perceptually uniform**, meaning Euclidean distances in LAB correlate with human perception of color difference. This is crucial for accurate boundary matching.

#### 2.2 Edge Extraction
```python
right_edge = piece_lab[:, -1, :]   # Last column (right boundary)
left_edge = piece_lab[:, 0, :]     # First column (left boundary)
bottom_edge = piece_lab[-1, :, :]  # Last row (bottom boundary)
top_edge = piece_lab[0, :, :]      # First row (top boundary)
```

#### 2.3 Gradient Computation (Sobel Operator)
```python
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradient
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradient
```

**Justification:** Gradient continuity helps detect edges that naturally continue across piece boundaries. Matching gradients ensures not just color similarity but also **structural continuity**.

---

### Step 3: Dissimilarity Matrix Computation

**Function:** `compute_dissimilarity_matrices(pieces, use_gradient=True)`

We compute two dissimilarity matrices:
- `h_dis[i,j]`: Cost of placing piece `j` to the **RIGHT** of piece `i`
- `v_dis[i,j]`: Cost of placing piece `j` **BELOW** piece `i`

#### Cost Function Components:

1. **LAB Color Distance:**
   ```python
   diff = right_edge - left_edge
   lab_cost = np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))
   ```

2. **Normalized SSD (NSSD):**
   ```python
   nssd = SSD / (std1 * std2 * edge_size)
   ```
   Normalizes the sum of squared differences by standard deviation, handling varying intensity levels.

3. **Gradient Continuity:**
   ```python
   grad_cost = np.mean(np.abs(grad_x_i[:, -1] - grad_x_j[:, 0]))
   ```

**Final Cost:**
```python
cost = lab_cost + 0.05 * nssd + 0.1 * grad_cost
```

---

### Step 4: Best Buddies Detection

**Function:** `find_best_buddies(h_dis, v_dis)`

The **Best Buddies** algorithm identifies **mutually best matches** — pairs of pieces that prefer each other as neighbors.

```python
# If piece A's best right neighbor is B, AND
# piece B's best left neighbor is A, then (A, B) is a horizontal best buddy pair

best_right = np.argmin(h_dis, axis=1)  # Best right neighbor for each piece
best_left = np.argmin(h_dis, axis=0)   # Best left neighbor for each piece

h_buddies = set()
for i in range(n):
    j = best_right[i]
    if best_left[j] == i:  # Mutual match!
        h_buddies.add((i, j))
```

**Key Insight:** Mutual best matches have **~90% reliability**. This is the foundation of our matching strategy — we trust these pairs and use them to seed the solution.

**Reference:** This approach is inspired by the "Best Buddies Similarity" concept from:
> Dekel, T., Oron, S., Rubinstein, M., Avidan, S., & Freeman, W. T. (2015). *Best-Buddies Similarity for Robust Template Matching*. CVPR.

---

### Step 5: A*-Inspired Region Growing (8×8 Puzzles)

**Function:** `solve_8x8_region_growing(pieces)`

For larger puzzles, we use an **A*-inspired region growing** algorithm:

#### 5.1 Seeding from Best Buddy Pairs
```python
# Start from the strongest (lowest cost) buddy pairs
buddy_pairs.sort()  # Sort by dissimilarity cost
for cost, p1, p2, direction in buddy_pairs[:10]:
    # Initialize grid with this buddy pair at center
    grid[3, 3], grid[3, 4] = p1, p2
```

#### 5.2 Frontier Expansion
```python
def get_empty_neighbors(grid, placed_set):
    """Get empty cells adjacent to placed pieces."""
    neighbors = set()
    for (r, c) in placed_set:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and grid[nr, nc] == -1:
                neighbors.add((nr, nc))
    return neighbors
```

#### 5.3 A* Selection: f(n) = g(n) + h(n)
```python
# Priority: 
# 1. Best Buddies (highest confidence)
# 2. Low Ambiguity (best_cost / second_best_cost)
# 3. Low Cost

candidates.sort(key=lambda x: (
    not x['is_buddy'],   # Buddies first (True < False)
    x['ambiguity'],      # Then low ambiguity
    x['cost']            # Then low cost
))
```

**Justification:** The ambiguity ratio helps avoid uncertain placements. If the best and second-best candidates have similar costs, the placement is risky.

---

### Step 6: Toroidal Shift Expansion

**Function:** Inside `solve_8x8_region_growing`

The relative arrangement may be correct, but the **absolute position** (which piece is at top-left) is unknown. We test all 64 possible shifts:

```python
for rs in range(grid_size):
    for cs in range(grid_size):
        shifted = np.roll(np.roll(arr, rs, axis=0), cs, axis=1)
        cost = compute_solution_cost(shifted, h_dis, v_dis)
        expanded.append({'arr': shifted, 'cost': cost, ...})
```

**Why Toroidal?** Row and column shifts preserve all internal boundaries while changing which piece appears at each position.

---

### Step 7: Border Variance Scoring

**Function:** `compute_border_score(arr, variances, grid_size)`

Original image borders typically have **lower variance** (often uniform backgrounds or edges). We score arrangements based on whether low-variance edges appear at the grid borders:

```python
# Top row pieces should have low top-edge variance
for c in range(grid_size):
    p = arr[0, c]
    score += variances[p]['T']  # Top edge variance

# Left column pieces should have low left-edge variance
for r in range(grid_size):
    p = arr[r, 0]
    score += variances[p]['L']  # Left edge variance
```

**Final Ranking:**
```python
# Lower score = better arrangement
expanded.sort(key=lambda x: (x['cost'] + 0.01 * x['border'], x['border']))
```

---

### Step 8: Solution Assembly

**Function:** `reassemble_image(pieces, arrangement, grid_size)`

```python
for r in range(grid_size):
    for c in range(grid_size):
        piece_idx = arrangement[r, c]
        result[r*piece_h:(r+1)*piece_h, c*piece_w:(c+1)*piece_w] = pieces[piece_idx]
```

---

## Algorithm Variants by Puzzle Size

### 2×2 Puzzles: Exhaustive Search with Voting

**Function:** `solve_exhaustive_2x2(pieces)`

With only 4 pieces (24 permutations), we can try all arrangements:

```python
from itertools import permutations

for perm in permutations(range(4)):
    arr = np.array([[perm[0], perm[1]],
                    [perm[2], perm[3]]])
    # Score with multiple criteria
```

**Voting Ensemble:** Multiple ranking strategies vote on the best permutation:
- Pure boundary cost (weight: 5)
- Pure border score (weight: 3)
- Boundary + border combo (weights: 2, 2)
- Corner score (weight: 1)

### 4×4 Puzzles: BFS + Greedy Hybrid

Uses Best Buddies graph with BFS placement, falling back to greedy for remaining pieces.

### 8×8 Puzzles: A*-Inspired Region Growing

Full A* algorithm with buddy seeding, ambiguity filtering, and toroidal expansion.

---
## Project Structure

```bash

        ├── data/
    │   ├── raw/
    │   │   └── GravityFalls/
    │   │       ├── correct/
    │   │       ├── puzzle_2x2/
    │   │       ├── puzzle_4x4/
    │   │       └── puzzle_8x8/
    │   └── solutions/
    │       ├── puzzle2x2/
    │       └── puzzle4x4/
    ├── notebooks/
    │   ├── 01_phase1_export.ipynb
    │   └── 02_final_solver.ipynb
    ├── Milestone_2.docx
    └── README.md

 
```

---

## Usage

### Running the Notebook

```python
# Run full test on all puzzles of a size
run_full_test(grid_size=2)  # Test all 2×2 puzzles
run_full_test(grid_size=4)  # Test all 4×4 puzzles
run_full_test(grid_size=8)  # Test all 8×8 puzzles

# Visualize the pipeline for a puzzle 4 class 4*4
visualize_pipeline_single(4,4)
```


---
## Design Decisions & Justifications

| Decision | Justification |
|----------|---------------|
| **LAB Color Space** | Perceptually uniform — color distances match human perception |
| **Sobel Gradients** | Captures edge structure continuity, not just color |
| **NSSD Normalization** | Handles varying brightness/contrast between pieces |
| **Best Buddies** | Mutual matches have ~90% reliability — use as confident anchors |
| **A* Region Growing** | Balances greedy speed with global optimization via ambiguity filtering |
| **Toroidal Shifts** | Solves the absolute position problem without additional computation |
| **Border Variance** | Exploits image border characteristics (often uniform) for final ranking |
| **Voting Ensemble (2×2)** | Combines multiple heuristics for robustness |

---
## Challenges & Solutions

### Challenge 1: Ambiguous Boundaries
**Problem:** Some piece pairs have similar costs, making the choice uncertain.  
**Solution:** Ambiguity filtering — prioritize placements where `best_cost / second_best_cost` is low.

### Challenge 2: Absolute Position
**Problem:** Greedy/BFS methods find relative arrangement but not which piece is at top-left.  
**Solution:** Toroidal shift expansion tests all possible absolute positions.

### Challenge 3: Local Minima
**Problem:** Greedy placement can get stuck in suboptimal arrangements.  
**Solution:** Multiple seeding points + Best Buddies as high-confidence anchors.

### Challenge 4: Border Detection
**Problem:** No explicit border pieces in some puzzles.  
**Solution:** Use edge variance — original image borders often have lower variance.
---

## Requirements

```
numpy
opencv-python
Pillow
scipy
matplotlib
```

---

## Author
Developed as part of Computer Vision coursework demonstrating classical image processing techniques for jigsaw puzzle assembly.

**Note:** This solver uses **NO AI/ML models** — all matching is done through classical computer vision algorithms (color analysis, gradient computation, graph algorithms).