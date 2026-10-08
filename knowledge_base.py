"""
Part 1 — Task 2: Nykaa Knowledge Base Documents.
Track: E-commerce & Retail (Nykaa).

Contains >= 12 required policy documents (2-5 sentences each) written specifically
for the Nykaa e-commerce domain covering all required topics.
"""

from typing import List, Dict

KB_DOCUMENTS: List[Dict[str, str]] = [
    {
        "doc_id": "KB-01",
        "topic": "return window by product category",
        "title": "Return Window Policy by Product Category",
        "content": (
            "Nykaa provides category-specific return windows to balance hygiene standards and customer satisfaction. "
            "Skincare, cosmetics, and personal fragrances must be initiated within 5 days of delivery and remain unopened with seals intact. "
            "Fashion apparel and footwear items are eligible for return within 15 days of delivery provided original tags are attached and items are unworn. "
            "Electronic beauty appliances can only be returned within 7 days if received with a verified manufacturing defect or transit damage."
        ),
    },
    {
        "doc_id": "KB-02",
        "topic": "COD refund timelines",
        "title": "Cash on Delivery (COD) Refund Timelines",
        "content": (
            "Refunds for Cash on Delivery (COD) orders cannot be remitted in cash by field delivery executives. "
            "Customers must submit their verified bank account IFSC details or UPI ID through the Nykaa app refund portal upon reverse pickup confirmation. "
            "Bank transfers via NEFT or IMPS are disbursed within 3 to 5 business days after our warehouse completes quality inspection. "
            "Alternatively, customers may opt for instant Nykaa Wallet store credits which are credited within 2 hours of pickup verification."
        ),
    },
    {
        "doc_id": "KB-03",
        "topic": "delivery SLAs",
        "title": "Standard Delivery Service Level Agreements (SLAs)",
        "content": (
            "Nykaa partners with premier logistics carriers to guarantee predictable delivery timeframes across India. "
            "Orders shipped to tier-1 metro cities (Mumbai, Delhi-NCR, Bengaluru, Chennai, Kolkata, and Hyderabad) are typically delivered within 2 to 3 business days. "
            "Tier-2 and tier-3 regional destinations have an SLA of 4 to 6 business days from the dispatch date. "
            "Deliveries to remote locations, northeastern states, and union territories may require 7 to 10 business days depending on terrain and weather conditions."
        ),
    },
    {
        "doc_id": "KB-04",
        "topic": "reverse-pickup eligibility",
        "title": "Reverse-Pickup Service Eligibility and Coverage",
        "content": (
            "Doorstep reverse pickup is complimentary for all eligible returns across more than 19,000 postal codes serviced by Nykaa logistics. "
            "The assigned courier executive attempts reverse pickup within 24 to 48 hours of return approval with three consecutive scheduled attempts. "
            "If your delivery pin code is outside active reverse-pickup coverage zones, you must self-ship the package via a registered courier. "
            "Nykaa reimburses courier self-shipping costs up to a maximum of INR 150 upon receiving a valid shipping bill."
        ),
    },
    {
        "doc_id": "KB-05",
        "topic": "warranty terms by category",
        "title": "Product Warranty Terms and Authenticity Guarantee",
        "content": (
            "Nykaa guarantees 100% genuine products sourced directly from authorized brand manufacturers and national distributors. "
            "Electronic beauty tools such as hair dryers, straighteners, and facial trimmers carry an official manufacturer brand warranty ranging from 1 to 2 years. "
            "Cosmetic consumables, personal care lotions, and makeup items do not have extended warranties beyond initial defect-free delivery. "
            "Customers should register their appliance warranty card with the respective brand service network using their Nykaa tax invoice."
        ),
    },
    {
        "doc_id": "KB-06",
        "topic": "order-cancellation policy",
        "title": "Order Cancellation Policy and Guidelines",
        "content": (
            "Customers can cancel an entire order or individual items directly from the 'My Orders' section as long as the status is 'Placed'. "
            "Once an order transitions to 'Shipped' or 'Dispatched' status, automated digital cancellation is completely locked in our warehouse management system. "
            "If cancellation is unavailable after shipment, customers must politely decline delivery when the courier executive arrives at their doorstep. "
            "For prepaid cancellations, the full transaction amount is refunded back to the source payment method within 24 to 48 hours."
        ),
    },
    {
        "doc_id": "KB-07",
        "topic": "loyalty-points redemption policy",
        "title": "Nykaa Privé Loyalty Points Redemption Policy",
        "content": (
            "Nykaa Privé reward points are earned on every verified purchase at the rate of 1 point per 100 rupees spent. "
            "Points can be redeemed during checkout where each reward point carries a monetary discount value of INR 1. "
            "A minimum cart value of INR 500 is required to apply reward points, and redemptions cannot exceed 50% of the total order value. "
            "Accumulated reward points remain valid for exactly 12 months from the date of credit before expiring automatically."
        ),
    },
    {
        "doc_id": "KB-08",
        "topic": "payment-failure/retry policy",
        "title": "Payment Failure and Cart Retention Policy",
        "content": (
            "If an online transaction fails but funds are debited from your bank account, your bank initiates an automated reconciliation. "
            "Debited funds for unconfirmed orders are reversed back to your source account within 24 to 48 business hours by your banking provider. "
            "Nykaa automatically reserves your selected shopping bag items for 15 minutes to allow seamless retry without losing inventory. "
            "If an order ID is still not generated after 2 hours, please share your bank reference number (UTR) with support."
        ),
    },
    {
        "doc_id": "KB-09",
        "topic": "size-exchange policy",
        "title": "Apparel and Footwear Size-Exchange Guidelines",
        "content": (
            "Nykaa offers one complimentary size exchange per item for fashion apparel and footwear selections. "
            "Size-exchange requests must be submitted within 7 days of delivery through the customer order dashboard. "
            "Exchange is strictly subject to alternative size availability in regional fulfillment centers; if unavailable, a refund is processed instead. "
            "Replacement items are dispatched immediately once the courier partner collects the original unworn article."
        ),
    },
    {
        "doc_id": "KB-10",
        "topic": "damaged-item claim process",
        "title": "Damaged or Tampered Item Claim Process",
        "content": (
            "If a shipment arrives damaged, leaked, or with a broken seal, an incident ticket must be logged within 48 hours of delivery. "
            "Customers must upload clear photographs or video footage of the outer shipping box, inner shipping label, and damaged product batch code. "
            "Our dedicated escalation quality team reviews and validates digital evidence within 24 to 48 hours. "
            "Once validated, customers receive an immediate free replacement shipment or an unconditional full refund."
        ),
    },
    {
        "doc_id": "KB-11",
        "topic": "international shipping restrictions",
        "title": "International Shipping and Cross-Border Restrictions",
        "content": (
            "Nykaa delivers to select international destinations including the UAE, Singapore, the United States, and the United Kingdom. "
            "Strict international aviation regulations prohibit the cross-border air transit of aerosol sprays, flammable perfumes, and nail paints. "
            "All cross-border consignments may be subject to local import customs duties and VAT taxes paid by the recipient upon arrival. "
            "Standard international delivery timelines span 10 to 18 business days depending on customs clearance speeds."
        ),
    },
    {
        "doc_id": "KB-12",
        "topic": "customer-support escalation matrix",
        "title": "Customer Support Escalation Matrix and Turnaround SLAs",
        "content": (
            "Nykaa provides a three-tiered escalation framework to ensure swift customer problem resolution. "
            "Level 1 consists of the automated AI support bot and frontline chat agents, offering immediate responses 24/7. "
            "If an issue remains unresolved, Level 2 Senior Support Supervisors intervene with a mandatory turnaround time of 12 hours. "
            "Level 3 Grievance Officers handle escalated compliance disputes within 24 to 48 business hours under an assigned official grievance reference ID."
        ),
    },
]


def get_all_documents() -> List[Dict[str, str]]:
    """Returns the full knowledge base corpus."""
    return KB_DOCUMENTS
