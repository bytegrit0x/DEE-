# DEE# Defensive Metrics & Algorithmic Framework

## 1. Core Defensive Metrics: Operational Definitions

### 1.1 Rotation Speed (RS)

**Precise Definition:** Time elapsed between a defensive recognition trigger event and the defender achieving optimal defensive positioning, measured in seconds.

**Operational Calculation:** 
$$RS = t_{position} - t_{trigger}$$

Where:
- $t_{trigger}$ = precise timestamp when a defensive need is identified through objective criteria
- $t_{position}$ = timestamp when defender reaches calculated optimal positioning

**Trigger Event Identification:**
Trigger events are programmatically identified using specific criteria:
1. Screen initiation: When offensive player's feet plant in screening position
2. Pass initiation: When ball leaves passer's hands
3. Drive initiation: When offensive player's hip angle changes >15° toward basket with simultaneous acceleration
4. Defensive call: Audio detection of specific defensive terminology ("screen," "help," etc.)

**Optimal Position Determination:**
Optimal defensive position $P_{opt}(t)$ is algorithmically calculated as:

$$P_{opt}(t) = \arg\min_{p \in \mathbb{R}^2} \left[ w_b \cdot d(p, B(t)) + w_o \cdot d(p, O_p(t)) + w_h \cdot d(p, H(t)) \right]$$

Where:
- $d(p, B(t))$ = distance from position $p$ to ball position $B(t)$
- $d(p, O_p(t))$ = distance from position $p$ to primary offensive assignment $O_p(t)$
- $d(p, H(t))$ = distance from position $p$ to help position $H(t)$
- $w_b, w_o, w_h$ = weights derived from regression analysis of elite defender positioning data

**Frame-level Position Achievement:**
Position achievement is detected when:
$$|P_{def}(t) - P_{opt}(t)| < \epsilon$$

Where:
- $P_{def}(t)$ = actual defender position at time $t$
- $\epsilon$ = threshold distance (set at 18 inches based on empirical analysis)

**Error Handling:**
To address noise and partial occlusion in tracking data, we implement:
1. Kalman filtering for trajectory smoothing
2. Linear interpolation for frames with <3 consecutive missing data points
3. Confidence scoring for each measurement, with low-confidence measurements flagged

### 1.2 Closeout Angle (CA)

**Precise Definition:** The angular deviation between a defender's approach vector and the calculated optimal approach angle that maximizes containment probability.

**Operational Calculation:**
$$CA = \min(|\theta_{approach} - \theta_{optimal}|, 360° - |\theta_{approach} - \theta_{optimal}|)$$

Where:
- $\theta_{approach}$ = measured angle of defender's movement vector
- $\theta_{optimal}$ = calculated optimal approach angle

**Potential Movement Direction Determination:**
Offensive player's potential movement directions are calculated through:

1. Historical directional tendency model:
   $$p(θ_{move} | P_o, P_d, court) = \frac{f(θ_{move} | P_o, P_d, court)}{\int_0^{2\pi} f(θ | P_o, P_d, court) dθ}$$
   
   Where:
   - $f(θ_{move})$ = frequency of movement in direction $θ$ given current positions and court location
   - $P_o$ = offensive player position
   - $P_d$ = defender position
   - $court$ = position on court (zone)

2. Current momentum vector analysis:
   $$\vec{v}_{momentum} = \frac{1}{n} \sum_{i=t-n}^{t} \vec{v}_i \cdot w_i$$
   
   Where:
   - $\vec{v}_i$ = velocity vector at frame $i$
   - $w_i$ = recency weight for frame $i$
   - $n$ = number of prior frames analyzed (set to 5)

3. Body orientation predictor:
   $$p(θ_{move} | θ_{hips}, θ_{shoulders}) = g(θ_{hips}, θ_{shoulders})$$
   
   Where $g$ is a learned function from elite offensive player data

**Optimal Angle Calculation:**
The optimal approach angle minimizes expected offensive advantage:

$$\theta_{optimal} = \arg\min_{\theta} \sum_{i=1}^{n} p(θ_i | P_o, P_d, court) \cdot EPA(θ_i, \theta)$$

Where:
- $p(θ_i | P_o, P_d, court)$ = probability of offensive movement in direction $θ_i$
- $EPA(θ_i, \theta)$ = expected points added if offense moves in direction $θ_i$ while defender approaches at angle $\theta$
- $EPA$ values derived from possession outcome database with >100,000 tracked possessions

### 1.3 Help Decision Quality (HDQ)

**Precise Definition:** A quantitative evaluation of help decision appropriateness based on spatiotemporal relationships, measured on a 0-100 scale.

**Operational Calculation:**
$$HDQ = 100 \cdot \left[ w_t \cdot S_t + w_d \cdot S_d + w_r \cdot S_r + w_v \cdot S_v \right]$$

Where:
- $S_t$ = timing score (0-1)
- $S_d$ = distance score (0-1)
- $S_r$ = recovery score (0-1)
- $S_v$ = value score (0-1)
- $w_t, w_d, w_r, w_v$ = weights determined through machine learning

**Timing Score Calculation:**
$$S_t = \max\left(0, 1 - \frac{|t_{actual} - t_{optimal}|}{t_{threshold}}\right)$$

Where:
- $t_{actual}$ = observed help decision timestamp
- $t_{optimal}$ = calculated optimal help timing
- $t_{threshold}$ = maximum acceptable timing deviation (set at 0.7 seconds)

**Optimal Help Timing:**
The optimal help timing $t_{optimal}$ is calculated as:

$$t_{optimal} = t_{threat} - \frac{d(P_d, P_{help})}{v_{max}}$$

Where:
- $t_{threat}$ = projected timestamp of primary defensive threat
- $d(P_d, P_{help})$ = distance from defender to help position
- $v_{max}$ = maximum defender velocity (calibrated per player)

**Distance Score Calculation:**
$$S_d = \max\left(0, 1 - \frac{|d_{actual} - d_{optimal}|}{d_{range}}\right)$$

Where:
- $d_{actual}$ = actual distance maintained to primary assignment
- $d_{optimal}$ = calculated optimal distance
- $d_{range}$ = acceptable distance range (position-specific)

**Optimal Distance:**
$$d_{optimal} = f(P_o, v_o, P_d, T_o, court)$$

Where:
- $P_o$ = offensive player position
- $v_o$ = offensive player velocity
- $P_d$ = defender position
- $T_o$ = offensive player threat level
- $court$ = court position
- $f$ = function derived from elite defender positioning data

**Recovery Score:**
$$S_r = p(recovery | d, v_d, a_{max})$$

Where:
- $p(recovery)$ = probability of successful recovery
- $d$ = distance to recovery position
- $v_d$ = current defender velocity
- $a_{max}$ = maximum defender acceleration (player-specific)

**Value Score:**
$$S_v = \frac{EPA_{without} - EPA_{with}}{EPA_{optimal}}$$

Where:
- $EPA_{without}$ = expected points added without help
- $EPA_{with}$ = expected points added with actual help
- $EPA_{optimal}$ = expected points added with optimal help

**HDQ Classification:**
The HDQ score maps to categorical assessments:
- 0-25: Poor
- 26-50: Adequate
- 51-75: Good
- 76-100: Excellent

### 1.4 Contest Effectiveness (CE)

**Precise Definition:** A quantitative measure of shot alteration capability without fouling, incorporating spatiotemporal positioning, measured as a percentage.

**Operational Calculation:**
$$CE = 100 \cdot \left[ w_c \cdot (FG\%_{uncontested} - FG\%_{contested}) + w_a \cdot A_{coverage} + w_t \cdot T_{score} \right] \cdot (1 - P_{foul})$$

Where:
- $FG\%_{uncontested}$ = field goal percentage when uncontested from shot location
- $FG\%_{contested}$ = field goal percentage when contested by similar contest profile
- $A_{coverage}$ = angular coverage score (0-1)
- $T_{score}$ = timing score (0-1)
- $P_{foul}$ = probability of committing a foul
- $w_c, w_a, w_t$ = importance weights determined through regression analysis

**Uncontested FG% Determination:**
Instead of relying on individual shooter data, we use:
1. Zone-specific league average FG% (13 court zones)
2. Shooter type classification (elite, above average, average, below average)
3. Shot type classification (catch-and-shoot, pull-up, floater, layup)

This creates a robust statistical baseline even for limited sample sizes.

**Angular Coverage Calculation:**
$$A_{coverage} = \frac{\int_{\theta_1}^{\theta_2} r_d(\theta) d\theta}{\int_{\theta_1}^{\theta_2} r_{max}(\theta) d\theta}$$

Where:
- $r_d(\theta)$ = radial profile of defender's contest at angle $\theta$
- $r_{max}(\theta)$ = maximum possible coverage at angle $\theta$
- $\theta_1, \theta_2$ = angular limits of shooting window

**Defender Radial Profile:**
$$r_d(\theta) = h_{vert} + l_{arm} \cdot \sin(\alpha_{arm}) \cdot \cos(\theta - \theta_{orient})$$

Where:
- $h_{vert}$ = vertical displacement
- $l_{arm}$ = arm length
- $\alpha_{arm}$ = arm angle from horizontal
- $\theta_{orient}$ = defender orientation angle

**Timing Score:**
$$T_{score} = \max\left(0, 1 - \frac{|t_{contest} - t_{release}|}{t_{window}}\right)$$

Where:
- $t_{contest}$ = contest apex timestamp
- $t_{release}$ = shot release timestamp
- $t_{window}$ = effective contest window (set at 0.2 seconds)

**Foul Probability:**
$$P_{foul} = \sigma\left(β_0 + β_1 \cdot d_{contact} + β_2 \cdot v_{closing} + β_3 \cdot \theta_{contact} + β_4 \cdot h_{contest}\right)$$

Where:
- $\sigma$ = sigmoid function
- $d_{contact}$ = minimum distance between defender and shooter
- $v_{closing}$ = closing velocity
- $\theta_{contact}$ = angle of potential contact
- $h_{contest}$ = contest height
- $β_i$ = coefficients derived from logistic regression on foul call data

### 1.5 Recovery Position (RP)

**Precise Definition:** A quantitative evaluation of a defender's position after initial defensive action, measuring readiness for subsequent actions on a 0-100 scale.

**Operational Calculation:**
$$RP = 100 \cdot \left[ w_p \cdot P_{score} + w_b \cdot B_{score} + w_e \cdot E_{score} \right]$$

Where:
- $P_{score}$ = position score (0-1)
- $B_{score}$ = balance score (0-1)
- $E_{score}$ = engagement score (0-1)
- $w_p, w_b, w_e$ = weights determined through machine learning

**Position Score:**
$$P_{score} = \max\left(0, 1 - \frac{d(P_d, P_{opt})}{d_{max}}\right)$$

Where:
- $P_d$ = actual defender position
- $P_{opt}$ = calculated optimal recovery position
- $d_{max}$ = maximum acceptable distance (set at 8 feet)

**Optimal Recovery Position:**
$$P_{opt} = \arg\min_{p \in \mathbb{R}^2} \sum_{j \in O} T_j \cdot w_j \cdot d(p, P_j) + w_b \cdot d(p, B)$$

Where:
- $O$ = set of offensive players
- $T_j$ = threat level of offensive player $j$
- $w_j$ = assignment weight for offensive player $j$
- $w_b$ = ball position weight
- $d(p, P_j)$ = distance from position $p$ to player $j$
- $d(p, B)$ = distance from position $p$ to ball

**Balance Score:**
$$B_{score} = f_{balance}(h_{com}, w_{stance}, \theta_{torso}, v_{lat})$$

Where:
- $h_{com}$ = height of center of mass
- $w_{stance}$ = stance width
- $\theta_{torso}$ = torso angle from vertical
- $v_{lat}$ = lateral velocity
- $f_{balance}$ = function derived from biomechanical analysis of elite defenders

**Balance Function:**
$$f_{balance} = \frac{w_{stance}}{h_{com}} \cdot \cos(\theta_{torso}) \cdot (1 - k \cdot |v_{lat}|)$$

Where $k$ is a scaling factor for velocity impact.

**Engagement Score:**
$$E_{score} = A_{coverage} \cdot R_{potential}$$

Where:
- $A_{coverage}$ = defensive coverage area
- $R_{potential}$ = reaction potential

**Coverage Area:**
$$A_{coverage} = \pi \cdot (r_{base} + r_{ext})^2$$

Where:
- $r_{base}$ = base coverage radius (position-specific)
- $r_{ext}$ = stance-dependent extension factor

**Reaction Potential:**
$$R_{potential} = \frac{v_{max} \cdot a_{max}}{1 + e^{-k \cdot (B_{score} - 0.5)}}$$

Where:
- $v_{max}$ = maximum velocity (player-specific)
- $a_{max}$ = maximum acceleration (player-specific)
- $k$ = scaling parameter
- $B_{score}$ = balance score

**RP Classification:**
The RP score maps to categorical assessments:
- 0-25: Poor
- 26-50: Adequate
- 51-75: Good
- 76-100: Optimal

## 2. Data Acquisition and Processing Pipeline

### 2.1 Video Capture Specifications

**Camera Requirements:**
- Side view: 120fps minimum, 1080p resolution
- Bird's eye view: 60fps minimum, 4K resolution
- Calibrated camera positions with known intrinsic/extrinsic parameters
- Synchronized timestamps across all cameras (±1ms tolerance)

**Preprocessing:**
1. Frame-level stabilization using court markers as reference points
2. Color normalization for consistent jersey identification
3. Resolution standardization to 1920×1080 (side) and 3840×2160 (overhead)
4. Temporal alignment of multi-camera feeds

### 2.2 Player Tracking System

**Computer Vision Pipeline:**
1. Background subtraction using dynamic median filtering
2. Player detection via YOLOv5 with specialized basketball player dataset (mAP > 0.92)
3. Jersey number recognition through OCR with team-specific color models
4. Player identification using jersey number + appearance consistency
5. Ball tracking using specialized spherical object detector with dynamic occlusion handling

**Pose Estimation:**
- Implementation: Modified OpenPose with basketball-specific training
- Keypoints: 17-point skeleton + 6 additional basketball-specific points
- Confidence threshold: Individual keypoint confidence > 0.7
- Temporal consistency enforcement through Kalman filtering
- Occlusion handling through keypoint prediction from temporal context

**3D Reconstruction:**
- Multi-view triangulation for accurate 3D positioning
- Error tolerance: ±3 inches XY-plane, ±2 inches Z-axis
- Court registration using homography transformation
- Player height normalization based on team roster data

### 2.3 Data Quality Control

**Automated Quality Checks:**
1. Tracking continuity verification (gaps < 5% of frames)
2. Velocity and acceleration boundary checking (flagging physically impossible movements)
3. Pose consistency validation (anatomically impossible configurations rejected)
4. Court boundary enforcement (players must remain within physical constraints)

**Manual Verification Protocol:**
1. Random sampling of 5% of processed possessions for human verification
2. Full review of any possession with >3 quality flags
3. Supervised correction of identified errors with feedback loop to improve processing

**Data Confidence Scoring:**
Each tracked frame receives a confidence score (0-1) based on:
- Tracking algorithm confidence
- Number of visible keypoints
- Consistency with physical movement models
- Resolution of player in frame
- Occlusion percentage

Low-confidence data (score < 0.7) is either:
1. Corrected through interpolation (if gap < 5 frames)
2. Flagged as low-confidence and incorporated with appropriate weighting
3. Excluded from analysis (if confidence < 0.4)

### 2.4 Feature Engineering

**Raw Features Extracted:**
1. **Spatiotemporal Coordinates:**
   - 3D position $(x, y, z)$ for all players and ball
   - Joint angles for key body parts (ankles, knees, hips, shoulders, elbows)
   - Head orientation vector
   - Center of mass estimation

2. **Kinematic Features:**
   - Velocity vectors (instantaneous and averaged over 3/5/10 frames)
   - Acceleration vectors
   - Angular velocities for key joints
   - Stride length and frequency

3. **Interaction Features:**
   - Pairwise distances between all players
   - Defensive assignment probabilities
   - Help potential metrics (distance/time to help position)
   - Screen proximity and angle

**Derived Features:**
1. **Spatial Pressure Maps:**
   $$P(x,y,t) = \sum_{i \in D} \frac{w_i}{(x-x_i)^2 + (y-y_i)^2 + c}$$
   
   Where:
   - $D$ = set of defenders
   - $w_i$ = defender effectiveness weight
   - $c$ = smoothing constant

2. **Voronoi-based Responsibility Maps:**
   $$V_i(t) = \{p \in \mathbb{R}^2 : d(p, P_i) \leq d(p, P_j) \cdot w_j \text{ for all } j \neq i\}$$
   
   Where:
   - $V_i(t)$ = area of responsibility for player $i$
   - $w_j$ = adjusted weight based on threat level and assignment

3. **Expected Defensive Impact:**
   $$EDI(t) = \sum_{p \in Points} p(score|p,D(t)) - p(score|p,D_{opt}(t))$$
   
   Where:
   - $p(score|p,D(t))$ = probability of scoring from point $p$ given actual defense
   - $p(score|p,D_{opt}(t))$ = probability given optimal defense

4. **Defensive Coordination Index:**
   $$DCI(t) = \frac{1}{|D|} \sum_{i \in D} \frac{d(P_i, P_i^{opt})}{d_{max}}$$
   
   Where:
   - $P_i^{opt}$ = optimal position for defender $i$
   - $d_{max}$ = normalization constant

## 3. Machine Learning Implementation

### 3.1 Model Architecture for Metric Optimization

**Two-Phase Approach:**
1. Supervised learning phase using expert-labeled examples
2. Reinforcement learning phase using possession outcomes

**Supervised Learning Models:**
- **Position Optimality:** Gradient-boosted decision trees (XGBoost)
  * Features: 85 spatial and kinematic variables
  * Target: Expert-rated position quality (1-5 scale)
  * Performance: RMSE < 0.4, R² > 0.85
  
- **Decision Quality:** Deep neural network with temporal convolution
  * Architecture: 3 temporal convolution blocks + 2 fully connected layers
  * Features: 120 spatiotemporal sequence features
  * Target: Expert decision quality ratings
  * Performance: 87% agreement with expert panel

- **Contest Effectiveness:** Ensemble of random forests and neural networks
  * Features: 67 contest-specific variables
  * Target: Shot FG% impact + expert ratings
  * Performance: Predicted FG% impact within ±3.5%

**Reinforcement Learning Refinement:**
- Environment: Possession simulator based on tracked play database
- Agent: Defensive position/decision policy
- Reward: -1 × Expected Points Added
- Architecture: Proximal Policy Optimization (PPO)
- Training: 500,000 simulated possessions
- Validation: Cross-entropy between model policy and elite defender actions

### 3.2 Weight Optimization Methodology

**Hierarchical Bayesian Optimization:**
1. Initial weights derived from expert consensus
2. Bayesian optimization to maximize correlation with possession outcomes
3. Position-specific weight adjustments through Gaussian process regression

**Metric-Specific Weight Learning:**
For each metric (e.g., Help Decision Quality):
$$\vec{w}_{HDQ} = \arg\max_{\vec{w}} \left[ ρ\left(HDQ(\vec{w}), EPA\right) - λ \cdot Ω(\vec{w}) \right]$$

Where:
- $ρ$ = Spearman rank correlation
- $EPA$ = possession outcome (expected points added)
- $Ω(\vec{w})$ = regularization term to prevent overfitting
- $λ$ = regularization strength

**Cross-Validation:**
- K-fold cross-validation (K=10) stratified by teams and game situations
- Hyperparameter tuning via Bayesian optimization
- Separate validation set (20% of data) held out entirely during development

**Interpretability Constraints:**
- Monotonicity constraints on key relationships
- Bounded weight ranges to ensure physical interpretability
- Non-negative weights for core components
- Maximum weight ratio constraints (no single factor can dominate)

### 3.3 Situation Matching Algorithm

**Feature Vector Construction:**
Each defensive situation is encoded as a high-dimensional feature vector:
$$\vec{f} = [f_{spatial}, f_{kinematic}, f_{contextual}, f_{personnel}]$$

Where each component contains:
- $f_{spatial}$ (32 dimensions): Relative positions, distances, angles
- $f_{kinematic}$ (28 dimensions): Velocities, accelerations, momentum
- $f_{contextual}$ (15 dimensions): Game state, possession time, score differential
- $f_{personnel}$ (10 dimensions): Player role encodings, physical attributes

**Similarity Function:**
$$S(D_{current}, D_{historical}) = \exp\left(-\frac{\sum_{i=1}^k γ_i(f_{current}^i - f_{historical}^i)^2}{2σ^2}\right)$$

Where:
- $γ_i$ = importance weight for feature $i$ (learned through gradient boosting)
- $σ$ = similarity scaling parameter (optimized through cross-validation)

**Efficient Retrieval:**
1. Locality-Sensitive Hashing (LSH) for initial candidate filtering
2. KD-tree spatial partitioning for regional similarity
3. Approximate Nearest Neighbor (ANN) search with hierarchical navigable small worlds

**Refinement Process:**
1. Retrieve top-100 candidate matches using approximate methods
2. Compute exact similarity scores for candidates
3. Return top-N matches that exceed similarity threshold τ (default τ = 0.85)
4. Cluster similar historical examples when appropriate

### 3.4 Validation Methodology

**Ground Truth Sources:**
1. Expert panel ratings (5 NBA/NCAA coaches per scenario)
2. Possession outcome correlation
3. Player tracking-derived defensive impact metrics
4. Win probability added in critical situations

**Validation Metrics:**
1. **Correlation with Expert Ratings:**
   - Pearson and Spearman correlations > 0.8
   - Mean absolute error < 0.5 on normalized scales
   - Inter-rater agreement (Krippendorff's α) > 0.75

2. **Predictive Performance:**
   - Possession outcome prediction accuracy
   - Shot quality impact prediction
   - Defensive breakdown prevention

3. **Reliability Analysis:**
   - Test-retest reliability with different video feeds
   - Inter-observer consistency when manual verification used
   - Sensitivity analysis to input perturbations

**Ablation Studies:**
- Component-wise ablation to quantify contribution of each feature group
- Perspective ablation (side-only vs. overhead-only vs. combined)
- Temporal context ablation (varying sequence length)

## 4. Computational Optimization

### 4.1 Real-time Processing Capabilities

**Parallelization Strategy:**
1. Frame-level parallelization across GPU cluster
2. Player-level parallelization within frames
3. Metric-level parallelization within analysis pipeline

**Hardware Specifications:**
- Processing server: 8× NVIDIA A100 GPUs, 64-core CPU, 512GB RAM
- Edge processing: NVIDIA Jetson Xavier NX at each camera
- Storage: 2PB high-speed NVMe storage array
- Network: 10Gbps dedicated connection

**Algorithmic Optimizations:**
1. Spatial partitioning to limit calculation scope
2. Multi-resolution processing (higher resolution near ball)
3. Temporal keyframing (full calculation at keyframes, interpolation between)
4. Early termination of unnecessary calculations
5. Incremental updating of derived metrics

**Latency Targets:**
- Full possession analysis: < 5 seconds
- Key metric updates: < 1 second
- Critical alert generation: < 200ms

### 4.2 Storage and Retrieval Optimization

**Data Compression:**
- Spatiotemporal trajectory compression using principal component analysis
- Keyframe-based skeletal pose storage with interpolation
- Variable precision based on feature importance

**Hierarchical Storage:**
1. Hot storage (NVMe): Recent games, frequent queries
2. Warm storage (SSD): Current season
3. Cold storage (HDD): Historical database
4. Archive (Tape): Raw video

**Query Optimization:**
- Materialized views for common query patterns
- Progressive loading (low-resolution first, details on demand)
- Pre-computed similarity matrices for historical comparisons
- Approximate query processing for exploratory analysis

**Caching Strategy:**
- LRU cache for recent queries
- Predictive pre-loading based on game state and user behavior
- Team-specific caching during active analysis sessions
