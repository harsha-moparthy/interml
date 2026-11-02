from expand import write
write(r'''
@@ 940
**Stationarity means the relevant statistical behavior does not change when the clock is shifted.** Strict stationarity requires the entire joint distribution of any finite collection of observations to remain unchanged under a common time shift. Weak stationarity requires only finite second moments, constant mean, and covariance determined by lag rather than calendar time. [1]

An IID sequence with a fixed finite-variance distribution satisfies both. A fixed-distribution IID sequence with infinite variance can be strictly stationary without satisfying weak stationarity's finite-variance requirement. Constant mean alone is insufficient: variability or dependence can still change over time.

**The tests start from opposite hypotheses.** ADF's null is a unit root under its specified regression setup. KPSS's null is stationarity around a level or deterministic trend, depending on the chosen option. Rejecting one null and failing to reject the other provides evidence to interpret, not a mathematical certificate about the process.

Choose lag length and trend terms deliberately. Short samples and structural breaks can make either test misleading or inconclusive. Examine plots, rolling behavior, and forecasting performance alongside test results. Differencing removes some forms of persistence but can also discard useful level information or introduce unnecessary dependence. The aim is a defensible model for the prediction task, not merely obtaining a preferred p-value.
@@ 941
**Random walks remember shocks, so unrelated series can appear to move together for long periods.** Each new innovation changes the level permanently. Two independent walks may both wander upward over a sample, producing a visually persuasive regression even though neither drives the other. [1]

Conventional regression t-statistics assume error behavior that this level-on-level regression may violate. Many observations along one slow excursion do not provide the same independent evidence as many unrelated draws. High R-squared and a small conventional p-value can therefore be spurious.

**Cointegration asks a stronger, different question.** If two nonstationary series share a stable long-run relation, some combination such as $Y_t-\beta X_t$ can be stationary. For example, let X be a random walk and let $Y_t=2X_t+u_t$ with stationary u. Both levels wander, but Y-2X fluctuates around a stable distribution.

Engle–Granger evaluates a fitted residual relation using appropriate critical values; Johansen handles multivariate relations. Independent random walks do not become cointegrated merely because their sample correlation is high. For any strategy evaluation, estimate using past data, account for selection among many pairs, test breaks, and include execution costs. A statistical relation is not by itself evidence of a profitable executable trade.
@@ 942
**ACF includes both direct and indirect relationships; PACF tries to isolate a lag's additional linear relationship.** In an AR(1) process, today's value depends on yesterday, which depends on the day before. Thus lag two can correlate with today even though the equation directly uses only lag one. [1]

For a stationary AR(1) with coefficient phi, its ACF decays like $\phi^k$. After accounting for the intervening lag, the ideal PACF is zero beyond lag one. More generally, a pure AR(p) has a PACF cutoff after p, while a pure MA(q) has an ACF cutoff after q.

**Use the patterns to propose models, not read an exact answer from noisy bars.** Finite samples fluctuate, mixed ARMA processes can make both functions tail off, and seasonal structure produces additional peaks. First determine whether differencing is appropriate, then inspect the resulting series and candidate lag structures.

ARIMA(p,d,q) applies d differences and models the differenced series with AR and MA terms. Compare plausible candidates with information criteria and rolling-origin forecasts, then inspect residual autocorrelation. Excess differencing can create artificial dependence and increase forecast uncertainty. The best-looking sample correlogram is not enough; the model must forecast reasonably under a time-respecting evaluation.
@@ 943
**Unpredictable direction can coexist with predictable magnitude.** After a volatile day, tomorrow may still be equally likely to rise or fall, but a large move may be more likely than during a quiet period. That produces weak return autocorrelation alongside dependence in absolute or squared returns. [1]

GARCH(1,1) updates conditional variance from a baseline, the last squared shock, and the last variance estimate: $\sigma_t^2=\omega+\alpha\epsilon_{t-1}^2+\beta\sigma_{t-1}^2$. Alpha controls reaction to new shocks; beta carries persistence forward.

**An illustrative calculation.** With omega=0.01, alpha=0.1, beta=0.8, previous squared shock 4, and previous variance 1, the next variance is 0.01+0.4+0.8=1.21. Without another large shock, the elevated variance can decay gradually rather than disappearing immediately. Units must match the return scale.

Under standard assumptions and alpha+beta below one, the unconditional variance is omega divided by one minus alpha minus beta. Heavy-tailed innovations and asymmetric responses to negative shocks may require extensions. Evaluate variance forecasts through likelihood, intervals, and forward validation. Forecasting risk magnitude does not imply forecasting return sign or earning a trading profit.
@@ 944
**A complex model only helps if its flexibility matches useful structure in the data.** Some forecasting benchmarks are dominated by trend and seasonality that a linear mapping captures well. A large attention model can overfit, optimize poorly, or fail to exploit temporal order effectively, allowing a carefully evaluated linear baseline to win. [1]

That result challenges a particular architecture and evaluation setup, not the possibility of transformer forecasting. Always compare against seasonal-naive and linear methods before attributing gains to model size.

**Patches change the unit of representation.** Instead of treating every scalar timestamp as a separate token, a patch groups a short local window. The model sees local shape within the patch and attends across fewer tokens. For a length-1024 sequence split into nonoverlapping patches of 16, attention sees 64 patch tokens rather than 1024 individual positions, though the full architecture adds other costs.

Foundation models add broad pretraining and different normalization, tokenization, and forecasting objectives. PatchTST, Chronos, TimesFM, and Moirai are not interchangeable implementations of one method. Compare horizons, covariate availability, tuning budgets, and pretraining overlap. Better tokenization does not eliminate regime shifts or unavailable future inputs. Evaluate the forecasts the application will actually consume using leak-free rolling splits.
@@ 945
**Quantization turns a continuous prediction problem into choosing among discrete symbols.** Uniform bins assign nearby numerical values to the same token. A categorical output can then represent uncertainty over bins and generate sampled futures, but it cannot distinguish values within a bin without an additional mechanism. [1] [2]

If the bin width is 0.2, nearest-bin rounding has error at most 0.1 for in-range values. Clipping a value outside the range can create much larger error. Narrow bins improve resolution but increase vocabulary size or reduce covered range.

Vector, lattice, or spherical quantization operates on richer latent representations rather than necessarily binning each raw return independently. It needs an encoder/codebook and a validated reconstruction path. Continuous projections retain numerical input precision; their output heads may predict a mean, a distribution, or a more flexible diffusion/flow-generated future.

**Match representation to the needed uncertainty.** A mean forecast can hide two plausible opposite outcomes. A discrete distribution can represent both but may poorly resolve rare extremes. Normalize using only history available at prediction time, and evaluate tail behavior, interval coverage, and cross-series scale changes. No tokenization family is inherently best for financial sequences. Compare reconstruction, forecasting, and decision quality under the same information constraints.
@@ 946
**A backtest must recreate what was knowable at the prediction time.** Suppose Monday's value is 10, Tuesday is missing, and Wednesday's value is 20. Linear interpolation fills Tuesday with 15 using Wednesday. A prediction supposedly made on Tuesday now contains information from the future. [1]

The same issue arises with centered rolling statistics or publication data revised later. A feature may look mathematically reasonable in a completed table while being impossible to compute live.

**Trailing windows are causal only under the right timestamp convention.** A 20-day z-score using today's close is available after that close, not before it. If the decision occurs before the close, shift the window accordingly. Publication and ingestion delays can require further shifts even when an event's nominal date is earlier.

Forward fill uses past values and can be causal, but stale observations may be misleading. Add age and missingness indicators and define expiration rules. Validate preprocessing with a streaming replay that exposes only records available at each decision time. Fit global scaling parameters on the training period only. A chronological train/test split does not rescue a feature pipeline that has already inserted future information into the training or test rows.
@@ 947
**Choosing today's survivors tells the historical model something about the future.** If the universe consists only of current index members, the backtest omits firms that failed, delisted, or left the index. It may also include successful firms before they were eligible for membership. [1]

Imagine two otherwise similar firms in an old year: one survives and joins a major index, while the other later fails. Selecting only today's members excludes the failure before the historical prediction is made. The simulated investor is receiving a future-survival filter unavailable at the time.

Use point-in-time membership and historical identifiers, including delisting outcomes and corporate actions. Verify that features themselves are also point-in-time: a survivorship-free price table is insufficient if sector labels, fundamentals, or liquidity filters use later information.

**The exact bias depends on the strategy.** It often makes historical results look better, but its sign is not a universal theorem. Document when securities entered the investable universe, how rebalances occurred, and what was tradable with the available information. Freeze the decision process for forward evaluation and include realistic costs. A high backtest return is not interpretable until universe construction and data availability are explicit.
@@ 948
**A correct direction forecast can be too small or too late to trade profitably.** A model may predict the mid-price will rise by two basis points, but buying at the ask and selling at the bid can consume more than that before fees or impact. Better statistical accuracy does not guarantee positive net value. [1]

An illustrative round trip earns four basis points gross, pays two in spread, one in fees, and two in impact and delay. Net return is minus one basis point. Repeating the trade more often increases turnover and repeatedly pays those costs.

**Execution changes the forecast's value.** A signal can decay while the order travels or waits in a queue. Larger orders may move prices or exhaust available liquidity. Passive orders can avoid immediate spread crossing but may not fill and can be adversely selected when they do.

The square-root impact relation is an empirical approximation with market and regime dependence, not an execution guarantee. Evaluate using event timestamps, plausible fills, latency, capacity, and sensitivity to cost assumptions. Compare the stronger model with the baseline at equal turnover or under a cost-aware policy. The relevant target is achievable net performance and risk, not only predictive accuracy on mid-price labels.
@@ 949
**Neutralization removes the component of a signal that lies in the span of chosen factor exposures.** If a model's score is mostly high for large companies, it may appear predictive because it is taking a size exposure rather than finding independent information. Regressing the score on size and other selected factors separates that linear component. [1]

Let F contain exposure columns and a contain scores for the same cross-section. The weighted least-squares fit is F times beta, and the residual is $a_\perp=a-F\hat\beta$. The normal equations give $F^TW a_\perp=0$: the residual is orthogonal to the included exposures under the selected weighting.

**A small example.** If scores equal twice a sector indicator plus a residual, projecting onto that sector column removes the common sector component. Adding an intercept can remove the cross-sectional mean as well. QR or SVD provides a stable way to perform the projection when exposures are correlated; explicit inverses can be fragile.

Neutralizing scores does not automatically make a portfolio neutral. Position sizing, constraints, and later transformations can reintroduce exposure. Nor does linear orthogonality remove nonlinear or future risk. Evaluate whether the removed component was unwanted risk or useful signal, and validate the resulting portfolio separately under time-respecting assumptions.
@@ 950
**Squared loss lets a few extreme residuals dominate the gradient.** Errors of 1 and 20 contribute 1 and 400 to squared loss. The larger error therefore receives 400 times the loss contribution, even if it arose from an unusual jump or a bad observation. [1]

Huber loss behaves quadratically near zero and linearly beyond a threshold. Its gradient stops growing proportionally with extreme error, reducing tail influence while retaining smooth behavior for ordinary residuals. That can improve robustness, but clipping the influence of genuine rare events may also hide risk important to the application.

**Other losses ask different questions.** Correlation losses emphasize cross-sectional association and ignore some scale information. A prediction vector multiplied by a positive constant has the same Pearson correlation but different probability or return calibration. Sharpe-inspired objectives combine average outcome and variability, yet can be unstable when estimated on small or unrepresentative batches.

Differentiable rank approximations add another approximation to the desired ordering metric. Compare objectives using forward validation, net outcomes after costs, tail behavior, and uncertainty. Do not choose a loss merely because its name resembles a preferred backtest statistic. The training objective, portfolio construction, and execution assumptions must together represent the intended decision problem.
''')
