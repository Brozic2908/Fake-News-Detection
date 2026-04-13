import os
import sys
import argparse
import re
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from models.fake_news_model import get_model


# =========================
# Helpers
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Inference for Fake News Detection")

    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path tới checkpoint .pt",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name, ví dụ: bert-base-uncased. Nếu không truyền sẽ lấy từ checkpoint.",
    )
    p.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Override max_len. Nếu không truyền sẽ lấy từ checkpoint hoặc mặc định 512.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="",
        help="Title bài báo cần dự đoán",
    )
    p.add_argument(
        "--text",
        type=str,
        default="",
        help="Text bài báo cần dự đoán",
    )
    p.add_argument(
        "--use_examples",
        action="store_true",
        help="Chạy thử 4 mẫu ví dụ có sẵn trong file",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda. Nếu bỏ trống sẽ tự chọn",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Số lớp, mặc định 2",
    )
    p.add_argument(
        "--real_label_id",
        type=int,
        default=0,
        help="ID của nhãn REAL. Mặc định = 0",
    )
    p.add_argument(
        "--fake_label_id",
        type=int,
        default=1,
        help="ID của nhãn FAKE. Mặc định = 1",
    )

    return p.parse_args()


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_input_text(title: str, text: str) -> str:
    title = clean_text(title)
    text = clean_text(text)

    if title and text:
        return f"{title} [SEP] {text}"
    return title or text


def load_checkpoint(ckpt_path: str, device: torch.device):
    print(f"[Inference] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def resolve_runtime_config(args, ckpt: dict):
    saved_args = ckpt.get("args", {})
    model_name = args.model or ckpt.get("model_name", saved_args.get("model", "bert-base-uncased"))
    max_len = args.max_len or saved_args.get("max_len", 512)
    dropout = saved_args.get("dropout", 0.3)
    return model_name, max_len, dropout


def build_model_and_tokenizer(model_name: str, dropout: float, num_classes: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        dropout_p=dropout,
        freeze_base=False,
    ).to(device)

    return model, tokenizer


def label_name_from_id(pred_id: int, real_label_id: int, fake_label_id: int) -> str:
    if pred_id == real_label_id:
        return "REAL"
    if pred_id == fake_label_id:
        return "FAKE"
    return f"CLASS_{pred_id}"


@torch.no_grad()
def predict_one(
    model,
    tokenizer,
    title: str,
    text: str,
    max_len: int,
    device: torch.device,
    real_label_id: int = 0,
    fake_label_id: int = 1,
) -> Dict:
    merged_text = build_input_text(title, text)

    encoded = tokenizer(
        merged_text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    logits = model(input_ids, attention_mask)
    probs = F.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    pred_label = label_name_from_id(pred_id, real_label_id, fake_label_id)

    result = {
        "pred_id": pred_id,
        "pred_label": pred_label,
        "confidence": float(probs[pred_id].item()),
        "probs": probs.detach().cpu().tolist(),
    }
    return result


def print_result(sample_name: str, title: str, result: Dict, real_label_id: int, fake_label_id: int):
    probs = result["probs"]

    real_prob = probs[real_label_id] if real_label_id < len(probs) else None
    fake_prob = probs[fake_label_id] if fake_label_id < len(probs) else None

    print("\n" + "=" * 80)
    print(f"Sample       : {sample_name}")
    print(f"Title        : {title[:150]}{'...' if len(title) > 150 else ''}")
    print(f"Prediction   : {result['pred_label']} (id={result['pred_id']})")
    print(f"Confidence   : {result['confidence']:.4f}")
    if real_prob is not None:
        print(f"Prob REAL    : {real_prob:.4f}")
    if fake_prob is not None:
        print(f"Prob FAKE    : {fake_prob:.4f}")
    print("=" * 80)


# =========================
# Example samples
# Bạn chỉ cần paste full text vào đây
# =========================
EXAMPLES: List[Dict] = [
    {
        "name": "TRUE_1",
        "title": "As U.S. budget fight looms, Republicans flip their fiscal script",
        "text": """WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a â€œfiscal conservativeâ€ on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBSâ€™ â€œFace the Nation,â€ drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense â€œdiscretionaryâ€ spending on programs that support education, scientific research, infrastructure, public health and environmental protection. â€œThe (Trump) administration has already been willing to say: â€˜Weâ€™re going to increase non-defense discretionary spending ... by about 7 percent,â€™â€ Meadows, chairman of the small but influential House Freedom Caucus, said on the program. â€œNow, Democrats are saying thatâ€™s not enough, we need to give the government a pay raise of 10 to 11 percent. For a fiscal conservative, I donâ€™t see where the rationale is. ... Eventually you run out of other peopleâ€™s money,â€ he said. Meadows was among Republicans who voted in late December for their partyâ€™s debt-financed tax overhaul, which is expected to balloon the federal budget deficit and add about $1.5 trillion over 10 years to the $20 trillion national debt. â€œItâ€™s interesting to hear Mark talk about fiscal responsibility,â€ Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican tax bill would require the  United States to borrow $1.5 trillion, to be paid off by future generations, to finance tax cuts for corporations and the rich. â€œThis is one of the least ... fiscally responsible bills weâ€™ve ever seen passed in the history of the House of Representatives. I think weâ€™re going to be paying for this for many, many years to come,â€ Crowley said. Republicans insist the tax package, the biggest U.S. tax overhaul in more than 30 years,  will boost the economy and job growth. House Speaker Paul Ryan, who also supported the tax bill, recently went further than Meadows, making clear in a radio interview that welfare or â€œentitlement reform,â€ as the party often calls it, would be a top Republican priority in 2018. In Republican parlance, â€œentitlementâ€ programs mean food stamps, housing assistance, Medicare and Medicaid health insurance for the elderly, poor and disabled, as well as other programs created by Washington to assist the needy. Democrats seized on Ryanâ€™s early December remarks, saying they showed Republicans would try to pay for their tax overhaul by seeking spending cuts for social programs. But the goals of House Republicans may have to take a back seat to the Senate, where the votes of some Democrats will be needed to approve a budget and prevent a government shutdown. Democrats will use their leverage in the Senate, which Republicans narrowly control, to defend both discretionary non-defense programs and social spending, while tackling the issue of the â€œDreamers,â€ people brought illegally to the country as children. Trump in September put a March 2018 expiration date on the Deferred Action for Childhood Arrivals, or DACA, program, which protects the young immigrants from deportation and provides them with work permits. The president has said in recent Twitter messages he wants funding for his proposed Mexican border wall and other immigration law changes in exchange for agreeing to help the Dreamers. Representative Debbie Dingell told CBS she did not favor linking that issue to other policy objectives, such as wall funding. â€œWe need to do DACA clean,â€ she said.  On Wednesday, Trump aides will meet with congressional leaders to discuss those issues. That will be followed by a weekend of strategy sessions for Trump and Republican leaders on Jan. 6 and 7, the White House said. Trump was also scheduled to meet on Sunday with Florida Republican Governor Rick Scott, who wants more emergency aid. The House has passed an $81 billion aid package after hurricanes in Florida, Texas and Puerto Rico, and wildfires in California. The package far exceeded the $44 billion requested by the Trump administration. The Senate has not yet voted on the aid. """,
        "expected": "REAL",
    },
    {
        "name": "TRUE_2",
        "title": "U.S. military to accept transgender recruits on Monday: Pentagon",
        "text": """WASHINGTON (Reuters) - Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday as ordered by federal courts, the Pentagon said on Friday, after President Donald Trumpâ€™s administration decided not to appeal rulings that blocked his transgender ban. Two federal appeals courts, one in Washington and one in Virginia, last week rejected the administrationâ€™s request to put on hold orders by lower court judges requiring the military to begin accepting transgender recruits on Jan. 1. A Justice Department official said the administration will not challenge those rulings. â€œThe Department of Defense has announced that it will be releasing an independent study of these issues in the coming weeks. So rather than litigate this interim appeal before that occurs, the administration has decided to wait for DODâ€™s study and will continue to defend the presidentâ€™s lawful authority in District Court in the meantime,â€ the official said, speaking on condition of anonymity. In September, the Pentagon said it had created a panel of senior officials to study how to implement a directive by Trump to prohibit transgender individuals from serving. The Defense Department has until Feb. 21 to submit a plan to Trump. Lawyers representing currently-serving transgender service members and aspiring recruits said they had expected the administration to appeal the rulings to the conservative-majority Supreme Court, but were hoping that would not happen. Pentagon spokeswoman Heather Babb said in a statement: â€œAs mandated by court order, the Department of Defense is prepared to begin accessing transgender applicants for military service Jan. 1. All applicants must meet all accession standards.â€ Jennifer Levi, a lawyer with gay, lesbian and transgender advocacy group GLAD, called the decision not to appeal â€œgreat news.â€ â€œIâ€™m hoping it means the government has come to see that there is no way to justify a ban and that itâ€™s not good for the military or our country,â€ Levi said. Both GLAD and the American Civil Liberties Union represent plaintiffs in the lawsuits filed against the administration. In a move that appealed to his hard-line conservative supporters, Trump announced in July that he would prohibit transgender people from serving in the military, reversing Democratic President Barack Obamaâ€™s policy of accepting them. Trump said on Twitter at the time that the military â€œcannot be burdened with the tremendous medical costs and disruption that transgender in the military would entail.â€ Four federal judges - in Baltimore, Washington, D.C., Seattle and Riverside, California - have issued rulings blocking Trumpâ€™s ban while legal challenges to the Republican presidentâ€™s policy proceed. The judges said the ban would likely violate the right under the U.S. Constitution to equal protection under the law. The Pentagon on Dec. 8 issued guidelines to recruitment personnel in order to enlist transgender applicants by Jan. 1. The memo outlined medical requirements and specified how the applicantsâ€™ sex would be identified and even which undergarments they would wear. The Trump administration previously said in legal papers that the armed forces were not prepared to train thousands of personnel on the medical standards needed to process transgender applicants and might have to accept â€œsome individuals who are not medically fit for service.â€ The Obama administration had set a deadline of July 1, 2017, to begin accepting transgender recruits. But Trumpâ€™s defense secretary, James Mattis, postponed that date to Jan. 1, 2018, which the presidentâ€™s ban then put off indefinitely. Trump has taken other steps aimed at rolling back transgender rights. In October, his administration said a federal law banning gender-based workplace discrimination does not protect transgender employees, reversing another Obama-era position. In February, Trump rescinded guidance issued by the Obama administration saying that public schools should allow transgender students to use the restroom that corresponds to their gender identity. """,
        "expected": "REAL",
    },
    {
        "name": "FAKE_1",
        "title": "Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.  2018 will be a great year for America! As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year. 2018 will be a great year for America!  Donald J. Trump (@realDonaldTrump) December 31, 2017Trump s tweet went down about as welll as you d expect.What kind of president sends a New Year s greeting like this despicable, petty, infantile gibberish? Only Trump! His lack of decency won t even allow him to rise above the gutter long enough to wish the American citizens a happy new year!  Bishop Talbert Swan (@TalbertSwan) December 31, 2017no one likes you  Calvin (@calvinstowell) December 31, 2017Your impeachment would make 2018 a great year for America, but I ll also accept regaining control of Congress.  Miranda Yaver (@mirandayaver) December 31, 2017Do you hear yourself talk? When you have to include that many people that hate you you have to wonder? Why do the they all hate me?  Alan Sandoval (@AlanSandoval13) December 31, 2017Who uses the word Haters in a New Years wish??  Marlene (@marlene399) December 31, 2017You can t just say happy new year?  Koren pollitt (@Korencarpenter) December 31, 2017Here s Trump s New Year s Eve tweet from 2016.Happy New Year to all, including to my many enemies and those who have fought me and lost so badly they just don t know what to do. Love!  Donald J. Trump (@realDonaldTrump) December 31, 2016This is nothing new for Trump. He s been doing this for years.Trump has directed messages to his  enemies  and  haters  for New Year s, Easter, Thanksgiving, and the anniversary of 9/11. pic.twitter.com/4FPAe2KypA  Daniel Dale (@ddale8) December 31, 2017Trump s holiday tweets are clearly not presidential.How long did he work at Hallmark before becoming President?  Steven Goodine (@SGoodine) December 31, 2017He s always been like this . . . the only difference is that in the last few years, his filter has been breaking down.  Roy Schulze (@thbthttt) December 31, 2017Who, apart from a teenager uses the term haters?  Wendy (@WendyWhistles) December 31, 2017he s a fucking 5 year old  Who Knows (@rainyday80) December 31, 2017So, to all the people who voted for this a hole thinking he would change once he got into power, you were wrong! 70-year-old men don t change and now he s a year older.Photo by Andrew Burton/Getty Images.",
        "text": """PASTE_FAKE_SAMPLE_1_TEXT_HERE""",
        "expected": "FAKE",
    },
    {
        "name": "FAKE_2",
        "title": "Drunk Bragging Trump Staffer Started Russian Collusion Investigation",
        "text": """House Intelligence Committee Chairman Devin Nunes is going to have a bad day. He s been under the assumption, like many of us, that the Christopher Steele-dossier was what prompted the Russia investigation so he s been lashing out at the Department of Justice and the FBI in order to protect Trump. As it happens, the dossier is not what started the investigation, according to documents obtained by the New York Times.Former Trump campaign adviser George Papadopoulos was drunk in a wine bar when he revealed knowledge of Russian opposition research on Hillary Clinton.On top of that, Papadopoulos wasn t just a covfefe boy for Trump, as his administration has alleged. He had a much larger role, but none so damning as being a drunken fool in a wine bar. Coffee boys  don t help to arrange a New York meeting between Trump and President Abdel Fattah el-Sisi of Egypt two months before the election. It was known before that the former aide set up meetings with world leaders for Trump, but team Trump ran with him being merely a coffee boy.In May 2016, Papadopoulos revealed to Australian diplomat Alexander Downer that Russian officials were shopping around possible dirt on then-Democratic presidential nominee Hillary Clinton. Exactly how much Mr. Papadopoulos said that night at the Kensington Wine Rooms with the Australian, Alexander Downer, is unclear,  the report states.  But two months later, when leaked Democratic emails began appearing online, Australian officials passed the information about Mr. Papadopoulos to their American counterparts, according to four current and former American and foreign officials with direct knowledge of the Australians  role. Papadopoulos pleaded guilty to lying to the F.B.I. and is now a cooperating witness with Special Counsel Robert Mueller s team.This isn t a presidency. It s a badly scripted reality TV show.Photo by Win McNamee/Getty Images.""",
        "expected": "FAKE",
    },
]


def main():
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.ckpt_path, device)
    model_name, max_len, dropout = resolve_runtime_config(args, ckpt)

    print(f"[Inference] Device   : {device}")
    print(f"[Inference] Model    : {model_name}")
    print(f"[Inference] Max len  : {max_len}")

    model, tokenizer = build_model_and_tokenizer(
        model_name=model_name,
        dropout=dropout,
        num_classes=args.num_classes,
        device=device,
    )

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print("[Inference] Checkpoint loaded successfully.")

    # Mode 1: chạy 4 ví dụ có sẵn
    if args.use_examples:
        correct = 0
        total = 0

        for sample in EXAMPLES:
            result = predict_one(
                model=model,
                tokenizer=tokenizer,
                title=sample["title"],
                text=sample["text"],
                max_len=max_len,
                device=device,
                real_label_id=args.real_label_id,
                fake_label_id=args.fake_label_id,
            )

            print_result(
                sample_name=f"{sample['name']} | expected={sample['expected']}",
                title=sample["title"],
                result=result,
                real_label_id=args.real_label_id,
                fake_label_id=args.fake_label_id,
            )

            if result["pred_label"] == sample["expected"]:
                correct += 1
            total += 1

        print(f"\n[Summary] Correct: {correct}/{total} | Accuracy on examples: {correct / max(total, 1):.4f}")
        return

    # Mode 2: chạy 1 mẫu tự nhập
    if not args.title and not args.text:
        raise ValueError("Bạn cần truyền --title hoặc --text, hoặc dùng --use_examples")

    result = predict_one(
        model=model,
        tokenizer=tokenizer,
        title=args.title,
        text=args.text,
        max_len=max_len,
        device=device,
        real_label_id=args.real_label_id,
        fake_label_id=args.fake_label_id,
    )

    print_result(
        sample_name="CUSTOM_INPUT",
        title=args.title,
        result=result,
        real_label_id=args.real_label_id,
        fake_label_id=args.fake_label_id,
    )


if __name__ == "__main__":
    main()