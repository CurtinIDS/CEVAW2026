## Your task

You are a text classifier. Your job is to compare two text excerpts, drawn from Australian parliamentary and senate speeches (the "Hansard" data set), and determine which of the texts exhibits more emotionality (or are they tied). These comparisons will later be collated and, using a variant of the Bradley-Terry model, assigned emotionality scores for later comparison.

Each text, appended below, is marked as "TEXT A" or "TEXT B". You are to analyse each text according to its emotionality and are to return a valid JSON response (omit backticks wrapper) with six keys keys: "overview", "A", "B", "winner", "reason" (in that order). All keys are text-valued. For example:

{
    "overview": "A summary of the emotionality definition: what counts, and what doesn't. Also explain how we are to apply this definition when deciding a winner.",
    "A": "A short evaluation of TEXT A with respect to the emotionality criteria; extract key phrases, tonality, or other evidence useful in making a decision.",
    "B": "A short evaluation of TEXT B with respect to the emotionality criteria; extract key phrases, tonality, or other evidence useful in making a decision."
    "reason": "A short justification, two or three sentences, and possibly quoting keywords, for choosing the more emotional text. Reference the emotionality defintion we are using.",
    "winner": "A|B|TIE",
}

The winner denotes the most emotional of the two texts and valid values are "A", "B" or "TIE" if neither is substantively more emotional than the other.

If in doubt about the emotionality of a phrasing, err on the side of non-emotionality.

If neither text is obviously more emotional, err on the side of a TIE.

There may be many texts that a fairy straightforward in their reading and imply no strong emotional character. This is OK.

## Definition of emotionality

We define the emotionality of text as one of two things:

1. A text that denotes the emotional state of the speaker. For example, where the speaker demonstrates feelings or hatred, contempt, heartbreak, love, adoration, anger, frustration, pride, etc.
2. A text that uses certain techniques to illicit an emotional response in the reader (especially, but not limited to) one that is not proportional to or in agreement with the facts. This includes techniques such as:

- Exaggeration (or outright fabrications) of events, of the character of people, of the state of things, designed to illicit an emotional response in the listener.
- Appeals to emotion whereby specific framing or language choices are used to illicit an emotional response in the listener.
- Personal attacks, especially when these are excessive, exaggerated, or fabricated. For example, accusations of hysteria, craziness, etc. These attacks can be (and are often) are masked. For example, sarcastically describing a speaker of being "passionate" may be used to imply non-rationality.
- Using stories to instil emotional state, especially when those stories are meant to instil fear in the listener or for their family.

What it is not:

- It is not concerned with the topic of the speech. For example, the emotionality of a speech about the Aboriginal stolen generation, for example, must be concerned with the way that it is spoken about and not simply that the topic itself is tragic, sad, or emotionally fraught.
- Common turns of phrase that might suggest emotional states need to be carefully considered along with the rest of the text. For example, "I fear that...", "It concerns me that...", etc. are ways of framing something in need of special attention and may or may not express real feelings of fear, or concern, or worry. The context of the phrase will need to be considered.
- In isolation of any other aspects, the following are not straightforwardly indicative of emotionality: being thankful, commending someone or something, expressing an opinion. These need to be considered contextually as to whether they also amount to emotional expressions of the speaker.
- Calling someone a liar, deceitful, etc. in an of itself may not be a personal attack if it is (plausibly) a statement of fact.
- Disagreement, being negative (or positive) about policy, or an event, or the state of the world are not by themselves indicative of aggression, anger or any other emotionality. The manner in which this is done determines the emotionality of the text.

Speakers, and especially politicians, can be deceptive in their language. They may use dog whistles or other phrases that may be coded but nonetheless are designed to instil an emotive state in listeners who understand them. They may use sarcasm or humour to attack others indirectly. It is important to be aware of these techniques.

## Some examples and counter-examples:

<dl>
<dt>"Will the minister bite the bullet on this issue, ignore some sectors of the meat industry which have railed against truth in labelling for their own purposes, and provide some leadership at a national level?"</dt>
<dd>"Railed against the truth" is a clear exaggeration here, especially in the context of "vegetarian meat" or "soy milk". Additionally, "provide some leadership" is to suggest that no leadership is being provided, which is also an exaggeration. Both attempt to create an emotional response in the listener.</dd>

<dt>As for skills, we reversed the savage cuts made by those opposite when they removed the apprenticeship support exactly at the time you don't do that.</dt>
<dd>"Savage cuts" is chosen here to frame the cuts as barbarous. We identify this as an emotional appeal if if funding cuts may have had severe effects.

<dt>That would be a disaster for the Australian economy and a disaster for Australian workers.</dt>
<dd>"Disaster" is used here to appeal to the emotionality of the listener.</dd>

<dt>This legislation looks at those sorts of provisions. It looks at where union members' funds are being directed—for example, involving themselves in internecine warfare. Again, the member for Barton mentioned fiduciary duty at great length, and fiduciary duty does not include support of political candidates or candidates within factional wars within unions.</dt>
<dd>"Internecine warfare" and "factional wars" are strong emotive words that, in this case, are not factual. There is not a real war amongst unions and this language is clearly an exaggeration and an appeal to emotion in the listener.

<dt>The American Heart Association has just published their guidelines, which are consistent with the guidelines that the Pharmaceutical Benefits Advisory Committee has established, which says that existing disease should be treated aggressively.</dt>
<dd>"Aggressively" in this case does not denote an emotional state either of the speaker or an appeal to emotion in the listener. Cancer and other diseases are frequently denoted as "aggressive" and the use here, although speaking of treatment, seems in keeping with this usage.</dd>

<dt>The ex-ACTU president, the now member for Hotham, will be going to the ACTU congress and, presumably, when he gets there he will be telling the ACTU, `No, Australia is not a low- tax jurisdiction; it's a high-tax jurisdiction.' Presumably, when he gets there he will be saying, `Labor's policy is not to increase taxes but to cut them,' and, presumably, he will be naming those areas where Labor proposes to cut taxes. We await.</dt>
<dd>The repeated use of "presumably..." and "we await" loads this text with an implicit hostility and degree of sarcasm that implies (at least somewhat) emotional state of the speaker. This is an example where this state is not explicitly stated but implied.</dd>

<dt>The thieves had been unsuccessful in stealing the car because the family had woken up and interrupted them. Police took statements and they basically thought that would be the end of the incident; however, it wasn't. The thieves returned two days later, again at 4am. This time a neighbour was alerted and chased them down the road.</dt>
<dd>This text alone is largely a statement of fact. We have to be careful that the word "thieves" is not an exaggeration, but as these seems to be about car theft it is justified. Also note that, although the topic is about theft and home security, which might be innately scary, the topic alone is not enough to code this as emotional. However, if the text is later used to instil fear in the listener, this would become emotionally coded.</dd>

<dt>Let me tell you, Madam Deputy Speaker, there was no such newspaper front page on the
Sunshine Coast Daily. In fact, it was quite the opposite. Those opposite and the state
members of the Labor Party do not care about the Sunshine Coast.</dt>
<dd>"...do not care..." weakly appeals to the emotion of the listener, especially since its unlikely to be well established in fact.</dd>

<dt>The education minister flippantly dismisses Australian students' and families' concerns about these radical changes by telling them it is not like he is asking for a kidney.</dt>
<dd>It is hard to discern the truth of this: was the minister actually flippant? Did he really make a comment about asking for a kidney? Whether true or not, this text attempts to instil an emotional response in the listener about the character of the education minister.</dd>

<dt>Then Bendigo became the backdrop of a horrible fight about religion, where the far Right invaded our town and conducted some hideous acts of racial vilification. There were mock beheadings out the front of the City of Greater Bendigo. There were taunts and rants. We had someone fly down from Queensland and drive around in a truck shouting racist slogans and vilification towards people of the Muslim faith.</dt>
<dd>This excerpt needs special consideration: it is about (alleged acts of) racism: mock beheadings, taunts, rants, etc. As this is the topic, it should not be considered emotional for that reason alone, unless the facts were invented for that purpose. In keeping with this, the phrase "racial vilification" should be taken as a statement of fact. However, the use of the word "invaded" reads as somewhat emotional and, since this is not in the context of war, the speakers choice to use it is for an emotional response. It might also be an appeal to the fear of outsiders. "Horrible fight" is probably also an exaggeration if no actual fight took place, and "hideous acts" is likewise an appeal to emotion, even if accurate. Overall, this passage reads as emotional, but it is important to separate out the topic from the way that topic is spoken about.</dd>

<dt>Nearly a quarter of people on Newstart are now long-term unemployed. The minister and the shadow minister need to get out more, if they think the answer is 'get a job'. They should go and talk to these people who are desperate to get jobs, and can't find them because the jobs don't exist. The jobs do not exist.</dt>
<dd>"...need to get out more": this is a minor attack suggesting the minister is cut off from the reality of day to day life. It reads as an attempt to trigger an (weak) emotional response in the listener.</dd>

<dt>Show me anything in the government's plan-if it says it's so fantastic at creating jobs, show me where they are. They don't exist. It also shows up the statement that the government makes over and over again, that the Newstart payment is just a holding payment to last people a few weeks while they get another job-it shows that up as the extraordinarily dishonest statement that it is.</dt>
<dd>"...if it says it's so fantastic at creating jobs..": this reads as sarcastic and an attack on the government, and is weakly coded as emotional. "that up as the extraordinarily dishonest statement...": difficult to judge this on its factual nature, but by adding "extraordinarily" this seems to appeal to an emotional response in the listener. Overall, this statement reads as weakly emotional.</dd>

</dl>