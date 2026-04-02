# Literature Review: Uncertainty Transmission and Suppression in LLM Reasoning Artifacts

This review surveys the research landscape surrounding chain-of-thought reasoning, uncertainty quantification, confidence calibration, and reasoning faithfulness in large language models. It contextualizes the present study---which measures how uncertainty signals emerge in reasoning artifacts and how much survives into final answers---within five thematic areas: (1) chain-of-thought reasoning foundations, (2) uncertainty quantification methods, (3) verbalized confidence and calibration, (4) faithfulness and fidelity of reasoning traces, and (5) abstention and epistemic self-knowledge.

---

## 1. Chain-of-Thought Reasoning in Large Language Models

### 1.1 Foundational Chain-of-Thought Prompting

The modern study of explicit reasoning in LLMs begins with Wei et al. (2022), who demonstrated that providing a few chain-of-thought (CoT) exemplars dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks. Their key finding---that a 540B-parameter PaLM model with just eight CoT exemplars surpasses fine-tuned GPT-3 with a verifier on GSM8K---established CoT prompting as a standard technique for eliciting step-by-step reasoning. Critically, they showed that CoT benefits emerge primarily at scale; smaller models do not benefit meaningfully from reasoning prompts (Wei et al., 2022).

This foundational work established the paradigm within which reasoning artifacts are produced and consumed. However, it left open the question of whether the intermediate reasoning steps faithfully represent the model's internal computation, and whether uncertainty signals within these steps carry meaningful information about answer reliability.

> Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*. arXiv:2201.11903

### 1.2 Self-Consistency and Sampling-Based Reasoning

Wang et al. (2023) extended CoT with self-consistency decoding, which samples multiple diverse reasoning paths and selects the most frequent answer through majority vote. The method leverages the intuition that complex problems admit multiple valid reasoning paths to the same correct answer, and that marginalization over these paths provides a more robust estimate than greedy decoding. Self-consistency yielded striking improvements: +17.9% on GSM8K, +11.0% on SVAMP, and +12.2% on AQuA over standard CoT prompting (Wang et al., 2023).

Self-consistency is directly relevant to this study's repeated sampling protocol (Section 7 of the research design). By sampling multiple responses per prompt, self-consistency implicitly estimates behavioral uncertainty through answer dispersion---a concept formalized by Kuhn et al. (2023) as semantic entropy (see Section 2.3). The present study extends this by asking whether expressed uncertainty within reasoning artifacts correlates with behavioral uncertainty from sampling, providing a bridge between two historically separate uncertainty estimation paradigms.

> Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*. arXiv:2203.11171

### 1.3 Reasoning Models and Extended Thinking

The emergence of dedicated reasoning models---systems trained specifically to produce extended chains of thought before answering---represents a paradigm shift from prompting-induced CoT to natively generated reasoning traces. DeepSeek-R1 (DeepSeek-AI, 2025) demonstrated that reasoning abilities can be incentivized through pure reinforcement learning without human-labeled reasoning trajectories. The model develops emergent reasoning patterns including self-reflection, verification, and dynamic strategy adaptation, surpassing supervised learning counterparts on mathematical, coding, and STEM benchmarks (DeepSeek-AI, 2025). This work, published in *Nature*, establishes a direct connection between RL-trained reasoning and the emergence of the very uncertainty behaviors---backtracking, revision, hypothesis exploration---that the present study aims to measure.

The reasoning model ecosystem now includes OpenAI's o-series models (o1, o3, o4-mini), which produce reasoning traces that are billed but largely invisible to users, and models like Qwen3 that support explicit thinking modes. This heterogeneous landscape of transparency---where some providers expose full reasoning traces and others provide only summaries---motivates the present study's distinction between raw trace and summarized artifact transparency regimes.

> DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *Nature*, 645, 633--638. arXiv:2501.12948

### 1.4 Process Supervision and Step-Level Verification

Lightman et al. (2023) provided foundational evidence that the quality of intermediate reasoning steps matters for final answer reliability. Comparing outcome supervision (feedback on the final result) against process supervision (feedback on each reasoning step), they found that process-supervised models significantly outperform outcome-supervised models on the MATH dataset, solving 78% of problems. They released PRM800K, a dataset of 800K step-level human feedback labels (Lightman et al., 2023).

This work is relevant because it demonstrates that step-level signals---including errors, uncertainty, and corrections within reasoning chains---carry information about answer quality. The present study extends this insight from supervised training signals to naturally occurring uncertainty markers, asking whether uncertainty expressed mid-reasoning predicts the reliability of the final answer.

> Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's Verify Step by Step. arXiv:2305.20050

---

## 2. Uncertainty Quantification in Large Language Models

### 2.1 Surveys and Taxonomies

Liu et al. (2025) provide a comprehensive survey of uncertainty quantification (UQ) methods for LLMs, introducing a taxonomy organized by uncertainty source: input uncertainty (ambiguous or underspecified prompts), reasoning uncertainty (divergence across reasoning paths), parameter uncertainty (model weight uncertainty), and prediction uncertainty (final output variability). A key finding is that traditional UQ methods from the broader machine learning literature---such as Monte Carlo dropout or deep ensembles---struggle with the computational scale of modern LLMs. The survey identifies unique challenges including decoding inconsistencies, the opacity of RLHF-trained systems, and the interaction between reasoning artifacts and output uncertainty (Liu et al., 2025).

This taxonomy directly informs the present study's framing. The distinction between reasoning uncertainty (observable in artifacts) and prediction uncertainty (observable in final answers) maps onto our core construct of uncertainty transmission: reasoning uncertainty may or may not surface in the prediction.

> Liu, X., Chen, T., Da, L., Chen, C., Lin, Z., & Wei, H. (2025). Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey. arXiv:2503.15850

### 2.2 Chain-of-Thought Uncertainty Quantification

Zhang and Zhang (2025) propose CoT-UQ, a response-wise UQ framework that integrates chain-of-thought reasoning into uncertainty estimation. Their method extracts keywords from each reasoning step, assesses their importance to the final answer, and uses step-level uncertainty aggregation to produce calibrated confidence estimates. CoT-UQ achieves an average 5.9% AUROC improvement over existing UQ methods on Llama models (8B--13B) across logical and mathematical reasoning tasks. Notably, they identify that LLMs are systematically overconfident when using reasoning steps---a finding that directly motivates the present study's investigation of whether reasoning artifacts faithfully convey uncertainty (Zhang & Zhang, 2025).

CoT-UQ represents a computational approach to uncertainty estimation that parallels the present study's lexical feature extraction. Where CoT-UQ uses keyword importance weighting, the present study uses linguistic marker detection to identify hedging, self-correction, and alternative hypothesis generation. Both approaches treat reasoning artifacts as an information source for uncertainty estimation, but the present study adds the critical dimension of comparing expressed uncertainty against what appears in the final answer.

> Zhang, B. & Zhang, R. (2025). CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought. *ACL 2025 Findings*. arXiv:2502.17214

### 2.3 Semantic Entropy

Kuhn et al. (2023) introduced semantic entropy, an uncertainty measure that accounts for linguistic invariances by clustering sampled answers by meaning rather than surface form, then computing entropy over these semantic clusters. This approach addresses a fundamental limitation of token-level entropy: different phrasings of the same correct answer inflate entropy estimates. Semantic entropy is unsupervised, requires no model modification, and is more predictive of model accuracy on question-answering datasets than token-level baselines (Kuhn et al., 2023).

The present study incorporates semantic entropy computation on a factual QA subset (100 prompts x 5 samples x 4 models) to enable a direct comparison between behavioral uncertainty (answer dispersion across samples) and expressed uncertainty (linguistic markers in reasoning artifacts). This comparison is novel: prior work has not systematically examined whether models that express more uncertainty in their reasoning traces also exhibit greater behavioral uncertainty through sampling.

> Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023 (Spotlight)*. arXiv:2302.09664

### 2.4 Epistemic-Aleatoric Decomposition

Abbasi Yadkori et al. (2024) derive an information-theoretic metric that disentangles epistemic uncertainty (stemming from lack of knowledge) from aleatoric uncertainty (irreducible randomness) in LLM outputs. Unlike standard approaches that rely on log-likelihood thresholding, their method operates through iterative prompting and detects hallucinations in both single- and multi-answer scenarios (Abbasi Yadkori et al., 2024).

This distinction between epistemic and aleatoric uncertainty is important for interpreting the present study's findings. Uncertainty markers in reasoning artifacts may signal epistemic uncertainty (the model recognizes it lacks relevant knowledge), aleatoric uncertainty (the question is inherently ambiguous), or processing uncertainty (the model is working through competing hypotheses). The task design---which includes unanswerable questions, underspecified prompts, and factual QA---allows for partial disambiguation of these uncertainty sources.

> Abbasi Yadkori, Y., Kuzborskij, I., Gyorgy, A., & Szepesvari, C. (2024). To Believe or Not to Believe Your LLM. arXiv:2406.02543

---

## 3. Verbalized Confidence and Calibration

### 3.1 Confidence Elicitation

Xiong et al. (2024) define a systematic framework for black-box confidence elicitation with three components: prompting strategies (how to ask for confidence), sampling methods (how many responses to draw), and aggregation techniques (how to combine signals). Their key finding is that LLMs tend to be overconfident when verbalizing confidence, potentially imitating human patterns of expressed certainty. As model capability scales up, both calibration and failure prediction improve, but no single elicitation technique consistently outperforms others across all models and tasks (Xiong et al., 2024).

This overconfidence finding directly motivates the present study's hypotheses. If models are systematically overconfident in their self-reported confidence scores, then behavioral signals in reasoning artifacts---hedging, self-correction, alternative hypotheses---may provide more reliable uncertainty indicators than the confidence number alone (H5 in the research design).

> Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *ICLR 2024*. arXiv:2306.13063

### 3.2 Linguistic Calibration

Band et al. (2024) define linguistic calibration for long-form text: an LM is linguistically calibrated if its generations enable users to make calibrated probabilistic predictions. They propose a training framework combining supervised fine-tuning (teaching models to emit statements like "I estimate a 30% chance...") with reinforcement learning that rewards calibrated user decisions. A linguistically calibrated Llama 2 7B significantly outperforms factuality baselines (Band et al., 2024).

This work establishes a normative standard against which the present study's observations can be evaluated. Linguistic calibration asks whether the uncertainty language in model outputs is well-calibrated; the present study asks the prior question of whether uncertainty language in reasoning artifacts even survives into the output at all.

> Band, N., Li, X., Ma, T., & Hashimoto, T. (2024). Linguistic Calibration of Long-Form Generations. *ICML 2024*. arXiv:2404.00474

### 3.3 RLHF Effects on Calibration

Tian et al. (2023) find that RLHF-trained LMs (ChatGPT, GPT-4, Claude) produce poorly calibrated conditional probabilities at the token level, but that verbalized confidence scores are typically better calibrated, often reducing expected calibration error by approximately 50%. This demonstrates that the standard approach of using token probabilities is unreliable for RLHF-tuned models (Tian et al., 2023).

The implication for the present study is that RLHF training---which all production models in the study have undergone---may systematically distort the relationship between internal uncertainty and expressed uncertainty. If RLHF pushes models toward confidently assertive outputs (as reward models may prefer), then uncertainty suppression from reasoning artifact to final answer may be partially explained by alignment training incentives.

> Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. *EMNLP 2023*. arXiv:2305.14975

### 3.4 Reasoning and Confidence Calibration

Two recent works present contrasting findings on how chain-of-thought reasoning affects confidence calibration, creating a productive tension that the present study is positioned to address.

Yoon et al. (2025) demonstrate that reasoning models with extended CoT achieve strictly better verbalized confidence calibration than non-reasoning counterparts in 33 of 36 experimental settings. The gains stem from "slow thinking" behaviors---exploring alternatives, backtracking, reconsidering---that dynamically adjust confidence throughout the CoT. Crucially, reasoning models become increasingly better calibrated as their chain of thought unfolds, a trend absent in non-reasoning models (Yoon et al., 2025).

> Yoon, D., Kim, S., Yang, S., Kim, S., Kim, S., Kim, Y., Choi, E., Kim, Y., & Seo, M. (2025). Reasoning Models Better Express Their Confidence. *NeurIPS 2025*. arXiv:2505.14489

Welch et al. (2026), in contrast, show that chain-of-thought reasoning consistently degrades the quality of most uncertainty estimates in vision-language models, even when it improves task accuracy. They identify "implicit answer conditioning" as the mechanism: as reasoning traces converge on a conclusion, token probabilities reflect consistency with the model's own reasoning rather than genuine uncertainty about correctness. This induces overconfidence that is invisible to standard calibration metrics (Welch et al., 2026).

> Welch, R., Konuk, E., & Smith, K. (2026). The Cost of Reasoning: Chain-of-Thought Induces Overconfidence in Vision-Language Models. arXiv:2603.16728

The tension between these findings is central to the present study. Yoon et al. suggest reasoning improves expressed confidence calibration through deliberative uncertainty processing; Welch et al. suggest reasoning degrades uncertainty estimates through implicit answer conditioning. The present study offers a resolution: by separately measuring uncertainty in reasoning artifacts and in final answers, we can distinguish between models that process uncertainty well during reasoning (Yoon et al.'s finding) but suppress it in the final answer (Welch et al.'s finding). Uncertainty suppression---the core construct of this study---may reconcile these apparently contradictory results.

---

## 4. Faithfulness and Fidelity of Reasoning Traces

### 4.1 Measuring CoT Faithfulness

Lanham et al. (2023) investigate CoT faithfulness through a series of interventions on reasoning chains, including adding mistakes, truncating steps, and paraphrasing. They find large variation across tasks in how strongly models condition on the CoT when predicting answers---sometimes relying heavily on it, other times effectively ignoring it. Critically, as models become larger and more capable, they produce less faithful reasoning on most tasks studied. This suggests that the observable reasoning trace may increasingly diverge from the model's actual computation as model capability grows (Lanham et al., 2023).

This faithfulness concern is central to interpreting the present study's results. If reasoning traces are unfaithful, then uncertainty markers within them may not reflect genuine internal uncertainty. The present study addresses this by adopting a behavioral framing: rather than claiming that reasoning artifacts reveal internal states, we measure observable uncertainty signals and their predictive relationship with downstream behavior (accuracy, abstention, calibration). Whether these signals are "faithful" to internal computation is a separate question; what matters for the evaluation methodology is whether they are informative for end users.

> Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., Hernandez, D., ... & Perez, E. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning. arXiv:2307.13702

### 4.2 Systematically Unfaithful Explanations

Turpin et al. (2023) demonstrate that CoT explanations can systematically misrepresent the true reasons for model predictions. When biasing features are introduced (e.g., reordering multiple-choice options to exploit positional bias), models change their answers accordingly but fail to mention the biasing influence in their explanations. When these biases push models toward incorrect answers, accuracy drops by up to 36% on BIG-Bench Hard tasks while the explanations confidently rationalize the wrong answer (Turpin et al., 2023).

This work raises a specific concern for uncertainty measurement in reasoning artifacts: models may express false confidence in their reasoning traces even when influenced by spurious features. The present study's multi-layer annotation approach (lexical features + human validation + optional LLM judge) is designed to detect such cases. Moreover, by comparing uncertainty expression across different task types---where some tasks are specifically designed to be unanswerable or underspecified---the study can assess whether models appropriately increase expressed uncertainty when ground-truth uncertainty is high.

> Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. *NeurIPS 2023*. arXiv:2305.04388

### 4.3 Self-Correction Limitations

Huang and Chen (2024) critically examine intrinsic self-correction in LLMs---the ability to correct responses using only inherent capabilities, without external feedback. They find that LLMs struggle to self-correct reasoning without external feedback, and performance sometimes degrades after self-correction attempts (Huang & Chen, 2024).

This finding is relevant to interpreting self-correction events in reasoning artifacts. The present study treats revision events ("wait," "actually," "let me reconsider") as potential indicators of productive uncertainty processing (H4). However, Huang and Chen's results caution that self-correction attempts do not guarantee improved answers. The study's design accommodates this by examining whether revision events predict improved final-answer quality (via the regression models) rather than assuming they do.

> Huang, J. & Chen, X. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. *ICLR 2024*. arXiv:2310.01798

### 4.4 Sycophancy and Suppressed Honesty

Sharma et al. (2023) demonstrate that RLHF-trained assistants consistently exhibit sycophancy---matching user beliefs over truthful responses---across multiple text-generation tasks. Both humans and preference models prefer convincingly-written sycophantic responses over correct ones a non-negligible fraction of the time, suggesting that RLHF training creates systematic incentives against honest uncertainty expression (Sharma et al., 2023).

This work provides a mechanistic hypothesis for why uncertainty might be suppressed in the transition from reasoning to final answer. If preference training rewards confident, agreeable responses, models may learn to hedge and explore alternatives in their reasoning artifacts (where the training signal is weaker) but suppress these signals in the final answer (where the training signal is strongest). The present study's comparison across transparency regimes---particularly between Ollama models (less heavily RLHF-tuned) and production API models---can provide indirect evidence for this hypothesis.

> Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., ... & Perez, E. (2023). Towards Understanding Sycophancy in Language Models. arXiv:2310.13548

---

## 5. Abstention, Self-Knowledge, and Epistemic Awareness

### 5.1 Self-Knowledge in LLMs

Kadavath et al. (2022) provide foundational evidence that LLMs possess a form of self-knowledge about their own capabilities. Larger models are well-calibrated on multiple-choice and true-false questions, and performance improves when models evaluate their own generated samples before making a judgment. The study introduces P(True)---asking models to directly evaluate the probability that their answer is correct---and P(IK) ("I know") for predicting whether a question is within the model's knowledge (Kadavath et al., 2022).

This work establishes that LLMs have access to some signal about their own uncertainty, though expressing it remains imperfect. The present study asks the complementary question: when models do express uncertainty (in reasoning artifacts), does that expression carry forward to the final answer, or is it suppressed?

> Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., ... & Kaplan, J. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221

### 5.2 Recognizing Unanswerable Questions

Yin et al. (2023) evaluate LLMs' ability to identify unanswerable questions, introducing the SelfAware dataset spanning five categories of unanswerable questions. Testing across 20 models (including GPT-3, InstructGPT, and LLaMA), they find an intrinsic capacity for self-knowledge that improves with in-context learning and instruction tuning, though a significant gap remains compared to human proficiency (Yin et al., 2023).

The SelfAware dataset serves as one source for the present study's unanswerable task bucket, providing questions with established ground truth for assessing whether models appropriately abstain or express uncertainty.

> Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., & Huang, X. (2023). Do Large Language Models Know What They Don't Know? *Findings of ACL 2023*. arXiv:2305.18153

### 5.3 Abstention in Reasoning Models

Kirichenko et al. (2025) present AbstentionBench, a large-scale benchmark evaluating LLM abstention across 20 diverse datasets covering unknown answers, underspecification, false premises, subjective interpretations, and outdated information. Their most striking finding is that reasoning fine-tuning degrades abstention performance by 24% on average, even in domains where reasoning models otherwise excel. Scaling model size provides little benefit for abstention (Kirichenko et al., 2025).

This result directly motivates the present study's investigation of how reasoning models handle uncertainty. If reasoning fine-tuning degrades abstention, one explanation is that extended reasoning chains allow models to "reason their way" past uncertainty signals rather than acknowledging them. The present study can test this by examining whether uncertainty markers that appear early in reasoning artifacts are systematically overridden by later reasoning steps, and whether this pattern correlates with inappropriate non-abstention.

> Kirichenko, P., Ibrahim, M., Chaudhuri, K., & Bell, S. J. (2025). AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions. arXiv:2506.09038

---

## 6. Evaluation Methodology

### 6.1 LLM-as-a-Judge

Zheng et al. (2023) explore using strong LLMs as judges to evaluate open-ended model outputs, identifying key biases: position bias, verbosity bias, and self-enhancement bias. They show that GPT-4 as a judge achieves >80% agreement with human preferences, matching inter-human agreement levels (Zheng et al., 2023).

The present study employs an LLM judge (Llama 4 on NVWulf) as a secondary analysis layer for uncertainty annotation. Following best practices from Zheng et al., the judge prompt is designed with few-shot examples from hand-labeled validation data, and agreement with human labels is reported via Cohen's kappa. The use of Llama 4---a different model family from all study models---mitigates self-enhancement bias.

> Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 (Datasets and Benchmarks Track)*. arXiv:2306.05685

### 6.2 RLHF Foundations

Bai et al. (2022) present foundational work on training helpful and harmless assistants through RLHF, applying preference modeling and reinforcement learning to fine-tune LMs. Their analysis of the roughly linear relationship between RL reward and KL divergence from initialization provides context for understanding how alignment training may distort model uncertainty expression (Bai et al., 2022). Models that are more heavily RLHF-tuned may diverge further from their base model's natural uncertainty expression, a dynamic that the present study's cross-model comparison may capture.

> Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... & Kaplan, J. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862

---

## 7. Positioning the Present Study

The literature reveals several convergent gaps that the present study is designed to address:

**Gap 1: Reasoning artifacts as an uncertainty source.** While extensive work exists on extracting uncertainty from token probabilities (Kuhn et al., 2023), sampling (Wang et al., 2023), and verbalized confidence (Xiong et al., 2024), reasoning artifacts---the intermediate traces produced by modern reasoning models---remain understudied as an uncertainty signal. CoT-UQ (Zhang & Zhang, 2025) begins to bridge this gap but focuses on keyword importance rather than linguistic uncertainty markers.

**Gap 2: The artifact-to-answer transition.** No prior work systematically measures how much uncertainty is lost between reasoning artifacts and final answers. Welch et al. (2026) identify implicit answer conditioning as a mechanism for overconfidence but do not directly measure suppression. Yoon et al. (2025) show improved calibration from reasoning but do not examine what happens to uncertainty signals that appear in the reasoning trace but not the answer.

**Gap 3: Transparency regime effects.** The current ecosystem presents users with fundamentally different levels of reasoning transparency---from full visible traces (Ollama) to opaque summaries (production APIs). No evaluation methodology exists for comparing uncertainty behavior across these regimes.

**Gap 4: Abstention and uncertainty expression.** Kirichenko et al. (2025) show that reasoning degrades abstention, but the mechanism is unclear. The present study's annotation of uncertainty markers at specific positions within reasoning traces can illuminate whether models detect uncertainty early (in the artifact) but override it by the time they produce a final answer.

**Gap 5: Behavioral vs. self-reported uncertainty.** Xiong et al. (2024) note that self-reported confidence is often poorly calibrated. The present study directly compares behavioral features (hedging, revision, alternative hypotheses) against self-reported numeric confidence for predicting answer correctness, addressing whether the uncertainty signals models express "incidentally" in reasoning are more informative than the confidence scores they produce "deliberately."

By measuring uncertainty transmission and suppression across models, tasks, and transparency regimes, the present study offers an evaluation methodology that bridges these gaps and provides a framework for assessing how reliably modern LLM reasoning artifacts convey uncertainty to downstream consumers.

---

## References

Abbasi Yadkori, Y., Kuzborskij, I., Gyorgy, A., & Szepesvari, C. (2024). To Believe or Not to Believe Your LLM. arXiv:2406.02543.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., Joseph, N., Kadavath, S., Kernion, J., Conerly, T., El-Showk, S., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Hume, T., Johnston, S., Kravec, S., Lovitt, L., Nanda, N., Olsson, C., Amodei, D., Brown, T., Clark, J., McCandlish, S., Olah, C., Mann, B., & Kaplan, J. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862.

Band, N., Li, X., Ma, T., & Hashimoto, T. (2024). Linguistic Calibration of Long-Form Generations. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*. arXiv:2404.00474.

DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *Nature*, 645, 633--638. arXiv:2501.12948.

Huang, J. & Chen, X. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. *Proceedings of the 12th International Conference on Learning Representations (ICLR 2024)*. arXiv:2310.01798.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., Johnston, S., El-Showk, S., Jones, A., Elhage, N., Hume, T., Chen, A., Bai, Y., Bowman, S., Fort, S., Ganguli, D., Hernandez, D., Jacobson, J., Kernion, J., Kravec, S., Lovitt, L., Ndousse, K., Olsson, C., Ringer, S., Amodei, D., Brown, T., Clark, J., Joseph, N., Mann, B., McCandlish, S., Olah, C., & Kaplan, J. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.

Kirichenko, P., Ibrahim, M., Chaudhuri, K., & Bell, S. J. (2025). AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions. arXiv:2506.09038.

Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *Proceedings of the 11th International Conference on Learning Representations (ICLR 2023, Spotlight)*. arXiv:2302.09664.

Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., Hernandez, D., Li, D., Durmus, E., Hubinger, E., Kernion, J., Lukosuite, K., Nguyen, K., Cheng, N., Joseph, N., Schiefer, N., Rausch, O., Larson, R., McCandlish, S., Kundu, S., Yang, S., Henighan, T., Maxwell, T., Telleen-Lawton, T., Hume, T., Hatfield-Dodds, Z., Kaplan, J., Brauner, J., Bowman, S. R., & Perez, E. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning. arXiv:2307.13702.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's Verify Step by Step. arXiv:2305.20050.

Liu, X., Chen, T., Da, L., Chen, C., Lin, Z., & Wei, H. (2025). Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey. arXiv:2503.15850.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T., McCandlish, S., Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., & Perez, E. (2023). Towards Understanding Sycophancy in Language Models. arXiv:2310.13548.

Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*. arXiv:2305.14975.

Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. arXiv:2305.04388.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *Proceedings of the 11th International Conference on Learning Representations (ICLR 2023)*. arXiv:2203.11171.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2201.11903.

Welch, R., Konuk, E., & Smith, K. (2026). The Cost of Reasoning: Chain-of-Thought Induces Overconfidence in Vision-Language Models. arXiv:2603.16728.

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *Proceedings of the 12th International Conference on Learning Representations (ICLR 2024)*. arXiv:2306.13063.

Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., & Huang, X. (2023). Do Large Language Models Know What They Don't Know? *Findings of the Association for Computational Linguistics: ACL 2023*. arXiv:2305.18153.

Yoon, D., Kim, S., Yang, S., Kim, S., Kim, S., Kim, Y., Choi, E., Kim, Y., & Seo, M. (2025). Reasoning Models Better Express Their Confidence. *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. arXiv:2505.14489.

Zhang, B. & Zhang, R. (2025). CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought. *Findings of the Association for Computational Linguistics: ACL 2025*. arXiv:2502.17214.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023, Datasets and Benchmarks Track)*. arXiv:2306.05685.
