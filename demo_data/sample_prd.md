# PRD: AI Loan-Operations Assistant for Merchant Lending v1

Owner: Product Team, Credit Operations (Lending)
Status: Draft for review
Target release: Q3 pilot with 50 field agents

## Background

Our company lends to small merchants across tier-2 and tier-3 cities. Credit
assessment today is manual: field agents collect bank statements, GST filings,
and bureau reports, and the ops team keys figures into the underwriting sheet
by hand. This process is slow, error-prone, and leads to mis-keyed repayment
figures that cost us in bad loans and slow disbursement. This PRD describes
version 1 of an AI Loan-Operations Assistant that lets the credit team assess
merchants faster using natural language and voice queries.

## Goals

- Reduce average time-to-credit-assessment from 45 minutes to under 12 minutes.
- Cut mis-keyed repayment data error rate by 40 percent within two quarters of launch.
- Increase weekly active usage of the digital credit file by 60 percent of the ops team.

## Non-goals

- Direct loan disbursement inside the assistant (disbursement stays in the lending core).
- Support for third-party co-lender underwriting in v1.

## Features

### F1. Credit data ingestion and verification pipeline

Allow admins to bulk upload bank statements as PDF or Excel files with repayment
data mapped to our internal merchant IDs. The system must validate rows, flag
duplicates, and report ingest errors in a review queue. Admins can approve or
reject batches before they publish to the live credit file. This feature depends
on the merchant taxonomy model being finalized by the data team.

### F2. Merchant and credit file search

Allow analysts to search the credit file by GSTIN, PAN, account number, or by
free-text description such as "kirana merchant with 18 months of clean repayment".
Search results must show repayment data, sanction conditions, risk grade, and
offered interest rate. Analysts can filter results by channel, risk grade, and
loan amount band. Analysts can export result sets to CSV for reconciliation
with their own systems.

### F3. Voice query in Hinglish

Support voice input so field agents can speak a mixed Hindi-English query like
"is wala merchant 18 percent interest pe eligible hoga" and receive the same
results as typed search. The assistant must transcribe with a word error rate
below 15 percent on our internal eval set and always confirm the interpreted
query on screen before running the search.
This depends on the merchant search being live.

### F4. Analyst dashboard with risk signals

Provide analysts a dashboard that shows recent searches, saved merchant
profiles, and risk notes for merchants they assess most often. Analysts can pin
frequent merchants to the top and see repayment alerts when a merchant they
assessed before misses an obligation. This depends on data ingestion.

### F5. Admin bulk upload review queue

Support admin review of ingested statement batches with a side-by-side diff
against the previous data version, validation error counts, and one-click
approve or reject. Admins can export an audit log of every publish action.
This depends on data ingestion.

### F6. Weekly adoption report for the field team

Allow admins and analysts to receive a weekly report covering adoption by
region, search success rate, voice usage share, and top failed queries.
Analysts can export the report data and schedule automatic delivery to the
field team distribution list.
This depends on data ingestion and the analyst dashboard.

### F7. Multilingual interface

Support English and Hindi interfaces with a language toggle, and localize
merchant category names and error messages. Translated risk notes remain in
English in v1.

### F8. Offline mode for low connectivity zones

Support a cached credit file snapshot so field agents in low connectivity zones
can run merchant searches offline, with queued result sync when connectivity
returns. This depends on merchant search.

## Success metrics

- Time-to-assessment median below 12 minutes.
- Mis-keyed repayment data error rate reduced by 40 percent.
- 60 percent weekly active ops adoption.
- Voice queries at least 20 percent of total searches by end of quarter.

## Risks and open questions

- Bank statement formats vary widely; parsing effort may be larger than estimated.
- Voice recognition accuracy for regional accents needs an eval set before launch.
- Legal review required before exporting merchant-level usage data to field teams.
