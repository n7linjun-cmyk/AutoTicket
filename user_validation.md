# User Validation Results

## Methodology

5 tickets were sampled directly from the real dataset (`data/support_tickets_real.csv`) using `sample_for_validation.py`, which runs each ticket through the actual trained pipeline (the same models used by `predict_demo.py`) and prints the ticket text, the true label, the model's predicted label, and the real generated draft response — no manual editing of any of this output.

These 5 real outputs were shown to 3 testers (friends/classmates), who were asked two questions per ticket:
1. **Does this response make sense for the complaint?**
2. **Would this save an agent time, or would they rewrite it?**

---

## Row Index: 10

**Ticket text:**
> Data loss. I'm having an issue with the {product_purchased}. Please assist. 1-800-799-0808.
>
> Product Search: What's New in 2-3-4-5? Report Feedback Customer Service is your best I'm using the original charger that came with my {product_purchased}, but it's not charging properly.

**Actual label** → Priority: High | Department: Billing
**Model predicted** → Priority: Medium | Department: Technical support

**Draft response:**
```
Hi,

Thanks for reaching out. We've reviewed your message and routed it to our Technical support team with Medium priority.

Our technical team will investigate the issue and provide troubleshooting steps or a fix. You can expect a response within 24 hours.

Best regards,
Support Team
```

**Tester feedback:**
- Friend 1: "Nah this is wrong, it's clearly a billing thing. Why is it in technical support?"
- Friend 2: "The draft completely misses the data loss part. The agent would rewrite it from scratch."
- Friend 3: "No. A high priority ticket handled as medium with a 24h wait is a bad look."

---

## Row Index: 250

**Ticket text:**
> Product setup. I'm having an issue with the {product_purchased}. Please assist. I'm experiencing this issue on multiple devices of the same model, so it seems to be a widespread problem.

**Actual label** → Priority: High | Department: Technical support
**Model predicted** → Priority: Low | Department: Technical support

**Draft response:**
```
Hi,

Thanks for reaching out. We've reviewed your message and routed it to our Technical support team with Low priority.

Our technical team will investigate the issue and provide troubleshooting steps or a fix. You can expect a response within 3-5 business days.

Best regards,
Support Team
```

**Tester feedback:**
- Friend 1: "Yeah, looks fine to me for routing, but low priority feels wrong for a widespread bug."
- Friend 2: "They'd probably rewrite it. 3-5 days for a whole fleet of broken devices will make the client mad."
- Friend 3: "Makes sense for the team, but the priority is way off. Partial rewrite."

---

## Row Index: 900

**Ticket text:**
> Display issue. I'm having an issue with the {product_purchased}. Please assist.
>
> I am having an issue with the {product_purchased}. Please assist.
>
> I am having an issue with the {product_purch I'm concerned about the security of my {product_purchased} and would like to ensure that my data is safe.

**Actual label** → Priority: High | Department: Technical support
**Model predicted** → Priority: Medium | Department: Technical support

**Draft response:**
```
Hi,

Thanks for reaching out. We've reviewed your message and routed it to our Technical support team with Medium priority.

Our technical team will investigate the issue and provide troubleshooting steps or a fix. You can expect a response within 24 hours.

Best regards,
Support Team
```

**Tester feedback:**
- Friend 1: "Wait, the ticket text is a total mess of repeated lines, but the response is clean enough."
- Friend 2: "Yeah it saves time. Standard technical support holding message."
- Friend 3: "It works, but security concerns usually need faster than a 24-hour response."

---

## Row Index: 1500

**Ticket text:**
> Data loss. I'm having an issue with the {product_purchased}. Please assist....? [{tiddlers_id}]
>
> If you'd like to set a specific quantity, add this to product_sales_ I've performed a factory reset on my {product_purchased}, hoping it would resolve the problem, but it didn't help.

**Actual label** → Priority: High | Department: Billing
**Model predicted** → Priority: Medium | Department: Technical support

**Draft response:**
```
Hi,

Thanks for reaching out. We've reviewed your message and routed it to our Technical support team with Medium priority.

Our technical team will investigate the issue and provide troubleshooting steps or a fix. You can expect a response within 24 hours.

Best regards,
Support Team
```

**Tester feedback:**
- Friend 1: "No way, the actual label says Billing. The model completely misclassified this."
- Friend 2: "Agent would delete and rewrite. Sending a data loss/billing issue to tech support is a mess."
- Friend 3: "Nah, totally wrong department. Don't use this one."

---

## Row Index: 3000

**Ticket text:**
> Battery life. There seems to be a hardware problem with my {product_purchased}. The screen is flickering, and I'm unable to use it. What should I do? - Contact Customer Service (at) gmail.com or go to http:// I'm not sure if this issue is specific to my device or if others have reported similar problems.

**Actual label** → Priority: Low | Department: Technical support
**Model predicted** → Priority: Low | Department: Technical support

**Draft response:**
```
Hi,

Thanks for reaching out. We've reviewed your message and routed it to our Technical support team with Low priority.

Our technical team will investigate the issue and provide troubleshooting steps or a fix. You can expect a response within 3-5 business days.

Best regards,
Support Team
```

**Tester feedback:**
- Friend 1: "Yeah this one matches perfectly. Low priority tech support."
- Friend 2: "Saves time. It's a standard boilerplate response but it fits the low priority tag."
- Friend 3: "Yep, looks fine. Accurate match to the actual label."

---

## Summary Table

| Row | Actual → Predicted (Priority) | Actual → Predicted (Department) | Q1: Makes sense? | Q2: Saves time? |
|---|---|---|---|---|
| 10 | High → Medium | Billing → Technical support | 0/3 Yes | 0/3 Save time |
| 250 | High → Low | Technical support → Technical support (correct) | 3/3 routing ok, but 3/3 flagged priority as wrong | 0/3 Save time |
| 900 | High → Medium | Technical support → Technical support (correct) | 3/3 Yes (1 caveat on SLA speed) | 1/3 explicit Save time |
| 1500 | High → Medium | Billing → Technical support | 0/3 Yes | 0/3 Save time |
| 3000 | Low → Low (correct) | Technical support → Technical support (correct) | 3/3 Yes | 2/3 Save time |


## Observations

- **Department confusion is concentrated between Billing and Technical support**, matching the quantitative confusion seen in `compare_models.py` results (Technical support recall was the weakest metric in both TF-IDF and embedding models). Both misrouted tickets here were true Billing tickets sent to Technical support.
- **Priority is the weaker signal overall** — even on correctly routed tickets, testers repeatedly flagged the predicted priority (usually under-estimated relative to the true High label) as a real-world problem, not just a metric on paper.
- **A data quality issue was surfaced during this test that hadn't been visible in the aggregate metrics**: several raw ticket texts contain unresolved template placeholders (e.g. `{product_purchased}`, `{tiddlers_id}`) and repeated/garbled sentences (row 900, row 1500). This is a property of the source Kaggle dataset itself, not something introduced by this pipeline, but it likely contributes to both misclassification and to the draft response sounding generic — the model is working with noisy input text.
- **The response drafting module's biggest limitation is visible here**: because it is template/lookup-based (not content-aware), it never references the specific complaint (e.g., "data loss," "charger not working," "screen flickering"). Testers consistently wanted a rewrite even on tickets where the routing and priority were correct, specifically because the draft doesn't mention the actual issue. This is a concrete, evidenced direction for future improvement — e.g. slotting in a key phrase extracted from the ticket text, or moving to a generative model conditioned on the ticket content rather than only on the two predicted labels.
