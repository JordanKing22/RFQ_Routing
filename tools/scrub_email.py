"""
Scrub real Outlook emails into fictional test fixtures for tests/emails/.

Real emails (.msg from classic Outlook, .eml from new Outlook, Outlook on the web, and Mac Mail, or
.zip files of them) go in private_emails/, which git ignores. This tool reads them with
mailfile.load and writes one fictional .eml per email into tests/emails/, plus an entry in
tests/emails/manifest.json with the lane it belongs in and "reviewed": false. Read every file it
writes before you commit it, then set "reviewed" to true.

    python tools/scrub_email.py "private_emails/RFQ 26-118.msg" --lane milling_3axis --rfq
    python tools/scrub_email.py private_emails/export.zip --lane orders --not-rfq --dry-run
    python tools/scrub_email.py private_emails/ --lane review --rfq --map "Globex=Hollis Gear"
    python tools/scrub_email.py "private_emails/*.msg" --lane turning --map "ourshop.com=mesaridgeprecision.com"

INPUT is files, folders, or wildcards (quoted wildcards work on Windows too). Every email read in
one run is scrubbed with what was learned from all of them, so run related emails together.
--lane review needs --rfq or --not-rfq; the other lanes imply it. --map-file picks another map.

What it replaces, the same way in every email and on every run (the real-to-fake map is kept in
private_emails/.scrub_map.json, so a rerun gives the same fakes):
    people      names in From/To/Cc and the same names in the subject and body: full names, first
                names alone, "Last, First", initials, possessives, names split across lines,
                greetings, sign-offs, signature blocks, quoted reply headers, "Mr. Smith", and a
                known first name followed by a surname anywhere in the text
    email       every address, mailto: links included (role mailboxes such as quotes@ keep their
                name, on a fake domain)
    domains     every domain and URL; paths and tracking queries are dropped
    companies   names ending in Inc, LLC, Corp, Co., Ltd, GmbH and the like, company lines in
                signatures, and names that match a sender's domain. Distinctive words are replaced
                and generic trade words (Precision, Aerospace, Machining) are kept, so the email
                still reads like the same kind of customer.
    phones      US and international numbers, extensions, fax; US fakes use the fictional 555-01xx
    addresses   street addresses, suites, PO boxes, "City, ST 12345" and "City, ST" lines
    --map       any other "Real=Fake" pairs, remembered in the map for later runs
Part numbers are kept (routing needs them) and always listed; --part-numbers replaces them.
Attachments are dropped (a drawing's title block cannot be scrubbed reliably) and listed by their
scrubbed names in an X-Scrubbed-Attachments header; --keep-attachments keeps them, unscrubbed.
Em and en dashes become plain hyphens (the repo allows none), and an email with no date gets a
fixed one, because tests/test_import.py checks both. Other headers (Received, Thread-Topic,
X-MS-Exchange-*, In-Reply-To) are not copied.

The review report lists every replacement (real -> fake, with counts) and a "check these" list of
anything in the output that still looks like a name, company, phone number, email address, or
street address. It prints real values, so it goes to the terminal only.

Standard library only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from email import utils as email_utils
from email.headerregistry import Address
from email.message import EmailMessage
from email.policy import default as _default_policy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "tests" / "emails"
DEFAULT_MAP = ROOT / "private_emails" / ".scrub_map.json"
LANES = ("milling_3axis", "milling_5axis", "turning", "itar", "orders", "review", "filtered")
# The lane usually settles whether an email is an RFQ; "review" holds both, so it needs a flag.
LANE_IS_RFQ = {"milling_3axis": True, "milling_5axis": True, "turning": True, "itar": True,
               "orders": False, "filtered": False}
EMAIL_EXTS = (".eml", ".msg", ".zip")
POLICY = _default_policy.clone(linesep="\n")
NEW_MANIFEST_ABOUT = ("Test emails for the mass upload tests. 'generated' files are fictional; "
                      "'scrubbed' files are real emails with every name, company, address, and phone "
                      "number replaced by tools/scrub_email.py. 'reviewed' says a person checked one.")


def _wordset(text: str) -> frozenset:
    return frozenset(w.lower() for w in text.split())


# ---------------------------------------------------------------------------------------------
# Built-in fictional values. None of these match a person, company, or street in the demo data
# (data/, shop_config.json), so a scrubbed email never reads like one of the demo's own
# customers; tests/test_scrub.py checks that.

FAKE_FIRST = """
Adeline Agnes Alden Alma Alton Ambrose Anders Ansel Arlo Astrid Atticus Aubrey Augustin Avery
Barnaby Beatrix Bennett Blaise Bram Cassius Cecily Celeste Cora Cyrus Dahlia Dashiell Delia
Desmond Dorian Edie Elias Eloise Emmett Enzo Estelle Ezra Fenna Fiona Flynn Gemma Gilbert Greer
Hattie Heloise Horace Idris Inez Ione Isadora Jonah Josie Juno Keaton Lachlan Larkin Leif Leona
Linus Lise Lorcan Lucian Mabel Magnus Malcolm Margo Matilda Maxine Milo Minerva Nell Niall Nico
Odette Orla Oscar Otis Percy Petra Philippa Quentin Quincy Ramona Reuben Rhea Roland Rosalind
Rufus Sabine Seamus Silas Solveig Stellan Sybil Thaddeus Thea Theo Una Ursula Vaughn Vera Wendell
Willa Winifred Xavier Yara Zelda Zora Anneke Bettina Calla Corwin Darcy Edgar Ellis Farrah Gunnar
Jolene Kiran Lorna Mirela Nadine Oren Perrin Rosamund Tove Ulric Wynn Yusuf Zane Aldo Benedikt
Clara Elsa Fabian Greta Hamish Ilse Jovan Kasimir Liesel Mara Nolan Olwen Piers Signe Teodor
Valentina Wilhelmina Cosima Lionel Maeve Anselm Birgit Cormac Dagny Eamon Freya Gwendolyn Ingo
""".split()

FAKE_LAST = """
Abernathy Ackroyd Aldridge Allingham Ambler Ansell Appleyard Ashdown Atherton Bagshaw Bamford
Barraclough Beckwith Bellingham Birkett Blenkinsop Botham Bracewell Bramhall Brassington
Brocklehurst Buckland Bunting Burbage Calvert Carrow Catterall Challis Chatwin Cheverton Clegg
Colley Cotterill Cragg Cresswell Crowther Dalby Darnell Daventry Dewhurst Dimmock Dorward
Duckworth Eastwood Eckersley Egerton Elsworth Emsley Entwistle Fairclough Farrow Fawcett Fenwick
Firth Fothergill Garside Gaskell Gilchrist Gledhill Goodall Greenhalgh Greaves Haigh Halliwell
Hargreaves Hawksworth Heseltine Hindle Holroyd Horsfall Hoyle Hulme Ingham Isherwood Jagger
Kershaw Kitchin Knowles Lambourne Lathom Leadbetter Lindley Lockwood Longbottom Lumb Marsden
Mellor Midgley Mosley Naylor Nuttall Oddie Ogden Openshaw Ormerod Pickles Pilling Prestwich
Pritchard Quarmby Ramsbottom Rawlinson Ridehalgh Rigby Roebuck Rushworth Sagar Satterthwaite
Scholes Shackleton Sharples Shuttleworth Sidebottom Sowerby Stansfield Starkey Sutcliffe
Tattersall Thistlethwaite Threlfall Tillotson Tordoff Townend Uttley Varley Wadsworth Wainwright
Walmsley Whiteley Whitworth Winterbottom Wolstenholme Yardley Ackerley Barlowe Crabtree Dunnock
Etherington Fairweather Gorton Hebblethwaite Illingworth Jowett Kilburn Lister Mottram Nettleton
Oldroyd Pennington Royle Stott Tatlock Unsworth Walsham Yeadon Brierley Cattermole Denholm
Fazackerley Gosling Hepworth Inskip Jolliffe Kenworthy Lightfoot Marchbank Normington Oxley
Pemberton Quayle Rossington Scatcherd Trelawney Underhill Vosper Wetherall
""".split()

# Company-name generator: an invented place-like stem replaces the distinctive words of a real
# name ("Acme Precision Inc." -> "Tarnwick Precision Inc."); a name made only of generic words
# gets a stem and a machine-shop customer trade ("Tarnwick Fluid Power").
COMPANY_STEMS = """
Tarnwick Brackenholt Culverdale Dunhallow Elderbrook Fenmoor Gorsefield Hallowmere Inchbrook
Jarrowby Kettlewick Larkhollow Marlbeck Netherby Penhallow Quarrington Rushmere Stonebeck
Umberfield Vellacott Wendmoor Yarrowdale Bellweather Coldharbour Deepwell Eastling Foxhollow
Greyholt Kingsmere Lindenhall Millbeck Orchardine Pennowick Silverdell Thistlewood Upfold
Verrowby Westerhope Yewdale Abbotsley Bramblecote Crowhurst Dallowmere Eskdale Fernhollow
Glenwick Hazelford Idlecombe Kirkhallow Loxley Merriton Nettlebed Oakhollow Pewsey Quorndon
Rookwood Shelburne Thornleigh Ullswick Varnhold Whinfell Yelverton Barrowdene Cragmoor Dovecote
Edgemoor Flintholm Gildersleeve Hollinwood Ivelford Kelderby Lathbury Mowbray Nantholt
Ottercombe Pendlebury Rivelin Sandholm Tanfield Wickersley Yarnbury Braithwell Cheswick Elmbridge
Frithwood Hartwell Keswold Lynmouth Marchmont Norbury Pickering Rydal Stainforth Tidewell
Wharfedale Amberlow Bexhollow Cobbleford Draycott Emberton Fulwick Grantley Hexworth Ivyholm
""".split()

COMPANY_TRADES = [
    ("Aerospace", "aero"), ("Aerostructures", "aerostruct"), ("Medical Devices", "med"),
    ("Surgical", "surgical"), ("Orthopedics", "ortho"), ("Robotics", "robotics"),
    ("Automation", "auto"), ("Fluid Power", "fluid"), ("Hydraulics", "hydraulics"),
    ("Pneumatics", "pneumatics"), ("Valve", "valve"), ("Pump Works", "pump"),
    ("Motion Controls", "motion"), ("Instruments", "inst"), ("Optics", "optics"),
    ("Photonics", "photonics"), ("Semiconductor Equipment", "semi"), ("Defense Systems", "defense"),
    ("Marine", "marine"), ("Energy Systems", "energy"), ("Oilfield Tools", "oilfield"),
    ("Motorsports", "motorsports"), ("Packaging Machinery", "packaging"), ("Test Systems", "test"),
    ("Controls", "controls"), ("Dynamics", "dynamics"), ("Industries", "ind"),
    ("Manufacturing", "mfg"), ("Engineering", "eng"), ("Technologies", "tech"),
    ("Gear Works", "gear"), ("Instruments Group", "instgroup"),
]

FAKE_STREETS = """
Millrace Ropewalk Bellows Tallow Cobblestone Lantern Pewter Tanyard Bobbin Tinsmith Wheelwright
Lampwick Gristmill Harness Saddlery Kilnworth Tannery Weir Sluice Brickyard Coachworks Forgeside
Smithy Cooperage Sawmill Flume Trestle Depot Switchyard Roundhouse Lamplighter Whetstone Plumbline
Drumlin Tinderbox Hopyard Brewhouse Ironbridge Wagonwheel Tollgate Stagecoach Dovetail Mortise
""".split()

# Fictional towns in real states.
FAKE_CITIES = [
    ("Harlow Springs", "OH"), ("Tarn Valley", "PA"), ("Brindlewood", "MI"), ("Calder Falls", "WI"),
    ("Dunmow Springs", "IN"), ("Elkhorn Crossing", "MN"), ("Fairhaven Mills", "NY"),
    ("Glenrock Junction", "CO"), ("Hadley Ridge", "TX"), ("Iverton", "IL"), ("Kestwold", "OR"),
    ("Lathom Bay", "WA"), ("Marrick", "NC"), ("Nettlefield", "SC"), ("Oxley Hollow", "TN"),
    ("Pellington", "GA"), ("Quarrow", "KY"), ("Rushford Flats", "KS"), ("Stavely", "MO"),
    ("Thornbury Heights", "AZ"), ("Ullendale", "UT"), ("Varrow Creek", "ID"), ("Wexley", "NV"),
    ("Yarrowby", "NM"), ("Ashtonvale", "CA"), ("Bramwick", "CT"), ("Cotterdale", "MA"),
    ("Dorwood", "NJ"), ("Edgerly", "VA"), ("Frayne", "AL"), ("Gorsey Point", "FL"),
    ("Hollins Ferry", "LA"), ("Inglewick", "IA"), ("Jessup Mills", "NE"), ("Kirkby Lake", "OK"),
    ("Linthwaite", "AR"), ("Merriton Falls", "NH"), ("Norrowby", "VT"), ("Oakenshaw", "ME"),
    ("Penbury", "DE"), ("Ridlington", "MD"), ("Scarcliffe", "WV"), ("Tadworth", "MS"),
    ("Upperby", "ND"), ("Venning", "SD"), ("Whitlow", "MT"), ("Yeadon Park", "WY"),
]

# Any area code with 555-0100 through 555-0199 is reserved for fiction.
FAKE_AREA_CODES = ("216 248 303 312 317 402 414 425 480 503 512 513 602 612 614 616 619 651 704 "
                   "713 714 719 720 763 801 810 813 816 847 858 864 901 916 919 937 949 952 970 "
                   "972 978").split()

# ---------------------------------------------------------------------------------------------
# Word lists that tell names from ordinary words.

# Words that often start a sentence or appear capitalized in RFQs, signatures, and disclaimers.
COMMON_WORDS = _wordset("""
a about above accept accepted account accounting accounts across action actual add added
additional address addressed adjust admin administrator advance advanced advice advise after
afternoon again against all allow also am america american an and annual another answer any
anyone anything anyway appreciate appreciated approval approve approved april are area around
as ask asked asking assembly assist assistance associate at attach attached attachment
attachments attention attn august authorized available away back balance bar base based batch be
because been before being below best better between bill billing blank blanket block board body
bold both bottom box brief bring budget build business but buy buyer buyers by call called can
cannot capability capacity care carrier case cell certificate certificates certification
certifications certified certs change changed changes charge chart check checked chief city
class clear click close closed code coming comment comments commercial company complete
completed confidential confidentiality confirm confirmation conformance contact contains
content contents contract contracts copy correct cost costs could country cover create credit
current currently customer customers cut daily data date dated day days dear december decimal
delete delivery department dept description design designed detail details did direct director
disclaimer discuss distribution division do document documents does done down draft drawing
drawings due during each early east edit effective either else email emails end engineer
engineering enough ensure enter entire error estimate estimated estimating etc evening even
every everyone everything exact example except exchange expected expedite expedited export
external extra fax february feel few field file files final finance find fine finish finished
first following follow followup for form forward forwarded free friday from full further future
general get getting give given glad go going good got great group guys had hand happy hard has
have having he head hear hello help her here hey hi him his hold holiday home hope hour hours
how however i if immediately important in inc include included includes including info
information inside inspect inspection instructions intended internal into invoice invoices is
issue issued issues it item items its itself january job jobs july june just keep kind kindly
know known large last late later latest lead least left legal less let letter level like limited
line lines link list little llc local location look looking lot lots made mail main make
manager many march material materials may maybe me mechanical meet meeting message messages
method might minimum mobile monday month months more morning most much must my name need needed
needs net new next nice no none normal north not note notes nothing notice notify november now
number numbers october of off offer office ok okay old on once one only open operations or
order ordered orders original other others our out outside over own page paid parts part pay
payment pdf pending per person phone please plus pm po point policy portal possible post
prefer president previous price priced pricing print printed privacy private process
procurement product production products program project proprietary provide provided purchase
purchasing qty quality quantities quantity question questions quick quickly quote quoted quotes
quoting rate rather re read ready really reason receipt receive received recipient recipients
reference regarding regards related release remit reply report request requested requesting
required requirement requirements reserved respond response rest review reviewed revised
revision rfq right rush safe sales same sample samples saturday say schedule scheduled second
section see send sender sending sent september service services set she ship shipment shipped
shipping short should show side sign signed since sir size small so some someone something
soon sorry south spec special specs standard start state states status still stock stop
subject submit such sunday supplier suppliers supply support sure system take team teams
technical tel terms test thank thanks that the their them then there these they thing things
think this those though through thursday time times title to today together tomorrow too top
total track tracking transmission tuesday type under unit united units unless until up update
updated upon urgent us usa use used user using vendor vendors very via view volume want wanted
was we website wednesday week weekly weeks welcome well were west what when where whether which
while who whole why will with within without work working works would write year years yes yet
you your yours
anodize anodized anodizing aluminum alloy steel stainless brass bronze copper titanium inconel
plastic nylon delrin acetal peek ultem polycarbonate passivate passivation plating plate plated
zinc nickel chrome chromate black hardcoat heat treat treated hardness tolerance tolerances
surface roughness thread threads threaded tapped hole holes bore bores diameter length width
height thickness weight dimension dimensions flatness bracket housing shaft pin bushing spacer
cover manifold fitting flange impeller valve cap nut bolt screw washer sleeve collar mount frame
panel rail arm lever clamp fixture insert nozzle adapter coupling hub wheel roller sensor
enclosure prototype prototypes first article fai cmm ppap coc cert mill milling milled lathe
turning turned swiss cnc edm grind grinding weld welding machining machined machine machines
axis model models step stp iges igs dxf dwg solidworks rev revs revision revisions assy
itar ear cui controlled restricted proprietary exempt usml eccn
individually bagged bag bags bulk critical characteristic characteristics hot rolled cold drawn
stress relieve relieved relief laser etch etched engrave engraved marking marked facility
facilities plant plants balloon ballooned sheet sheets gauge gage rough roughing finishing
thanksgiving christmas holidays warehouse dock hours tooling setup setups fixturing lot lots
cheers regards sincerely respectfully cordially thx tia obrigado obrigada gracias saludos
atentamente merci cordialement danke grazie saluti mfg
""")

# Generic words in company names: kept as they are, because they say what kind of customer it
# is (and alone they identify no one).
INDUSTRY_WORDS = _wordset("""
precision machine machining machined machinery manufacturing mfg mfr industries industrial
engineering engineered technologies technology tech systems solutions products aerospace aero
aviation avionics defense defence medical devices device surgical dental orthopedic
orthopedics robotics automation controls control fluid power hydraulic hydraulics pneumatic
pneumatics valve valves pump pumps motion instruments instrument optics optical photonics
semiconductor equipment energy marine motorsports motorsport racing packaging test testing tool
tools tooling fabrication fabricators fab metal metals metalworks works group holdings
enterprises international global usa america american national general united advanced
superior quality custom labs laboratories research dynamics components parts supply services
service design designs electric electrical electronics plastics composites castings casting
forge forging forgings foundry welding gear gears bearing bearings fastener fasteners sensors
scientific space satellite rail automotive motors motor turbine turbo turbomachinery oil gas
solar nuclear water environmental agricultural agriculture farm food brewing firearms arms
tactical ordnance associates partners consulting company corp corporation inc incorporated llc
ltd limited co gmbh plc and of the & north south east west central pacific atlantic midwest
southwest northwest southeast northeast western eastern northern southern mountain valley
coast bay lakes innovations innovative integrated applied allied universal standard premier
pro elite apex summit pioneer frontier liberty eagle star mfg. machine-works shop shops
usinagem usinage mecanica mecanique mecanizados maquinados industria industrias industrie
maschinenbau fertigung technik zerspanung metalurgica
""")

# Job titles and departments; a signature line made of these is not a name.
TITLE_WORDS = _wordset("""
buyer buyers purchasing procurement manager managers engineer engineers engineering director
president vp vice ceo coo cfo cto cio owner founder cofounder partner principal specialist
coordinator planner analyst sourcing supply chain quality senior sr jr junior lead assistant
associate administrator admin executive officer representative rep account accounts sales
customer service operations production program project commodity strategic supplier
development design mechanical manufacturing process technician supervisor foreman estimator
inside outside regional national general contracts contract materials material logistics
shipping receiving office team department dept division head chief controller payable
receivable buyer/planner inspector inspection machinist programmer scheduler expediter
counsel attorney secretary treasurer marketing business mgr eng qa qc hr it ops
""")

# Common English words that are also names. A real name made of one of these is replaced
# anywhere when it sits next to the rest of the name, but alone only where the context says it
# is a name ("Hi Mark", "Mark," on its own line, "Mark's", "Mr. Price"), so "mark the parts" and
# "unit price" survive.
NAME_WORDS = _wordset("""
mark will bill rich frank grant hunter chase lane case hill wood stone bond ball cook price
miller turner mason baker carpenter parker fisher porter wells banks street field ford king
knight page ward rice bell cross day dean drew faith hope joy grace glen dale gene guy jack jean
pat ray rod sue van wade west moore more best love little short sharp strong swift nash cash
gold silver steel iron bolt black white brown green gray grey young long rose may june april
august summer autumn winter dawn eve sky reed cole hale holly ivy iris ruby amber crystal pearl
jade penny sandy rusty art don gus nick pierce cliff bud chip chuck buck duke earl major judge
bishop prince sterling brook heath moss marsh fields hart park parks rivers waters woods carter
cooper fowler gardner hand head salt sage basil clay flint forrest wolf fox crane swan finch
drake robin jay lark hawk lamb bass pike trout fish bush tree branch root berry rock ridge law
rule box bank mills mill burns fry frost snow storm rain tide beach shore coast port gates wall
tower hall castle church noble wise bright sweet merry blue red olive hazel heather daisy lily
violet poppy fern laurel myrtle rowan ash oak birch elm cedar pine willow aspen maple joe max
sam lou tim tom ron roy val kit bob sal al ed jo dot ty lee
""")

# Frequent real first names. A capitalized surname after one of these is taken as a person
# even when the name is in no header ("please call Dennis Okafor"), and one left alone in the
# output goes on the "check these" list.
COMMON_FIRST_NAMES = _wordset("""
james john robert michael william david richard joseph thomas charles christopher daniel matthew
anthony donald steven paul andrew joshua kenneth kevin brian george edward ronald timothy jason
jeffrey ryan jacob gary nicholas eric jonathan stephen larry justin scott brandon benjamin samuel
gregory alexander raymond patrick dennis jerry tyler aaron jose adam henry nathan douglas zachary
peter kyle walter ethan jeremy harold keith christian roger noah gerald carl terry sean austin
arthur lawrence jesse dylan bryan jordan billy bruce albert willie gabriel logan alan juan wayne
ralph randy eugene vincent russell elijah louis bobby philip johnny mary patricia jennifer linda
elizabeth barbara susan jessica sarah karen nancy lisa betty margaret sandra ashley kimberly
emily donna michelle dorothy carol amanda melissa deborah stephanie rebecca sharon laura cynthia
kathleen amy shirley angela helen anna brenda pamela nicole emma samantha katherine christine
debra rachel catherine carolyn janet ruth maria diane virginia julie joyce victoria olivia kelly
christina lauren joan evelyn judith megan cheryl andrea hannah martha jacqueline frances gloria
ann teresa kathryn sara janice alice madison doris abigail julia judy denise marilyn beverly
danielle theresa sophia marie diana brittany natalie isabella charlotte alexis kayla mike dave
jim steve chris dan matt tony ben rob rick greg jeff ken jon andy brad phil kate katie liz beth
jen jenny pam deb barb cathy kathy becky kim tina terri chad todd troy derek travis shawn cody
luis carlos jorge miguel pedro raul ricardo alejandro fernando javier sergio manuel mario ana
rosa carmen lucia sofia priya raj rahul amit anil sanjay vijay deepak wei ming hui jun hiroshi
kenji yuki olga ivan dmitri sergei hans klaus pierre marco luca giuseppe jane jill janet joanne
kristen kristin heidi wendy tammy tracy stacy dawn connie gail lori sheila tonya vanessa monica
erin kara krista leah allison brooke caitlin colleen courtney haley jenna kelsey lindsay molly
shannon tara whitney craig curtis dale darren dean dustin edwin frederick glenn gordon howard
jeremiah joel jared leonard lucas marcus martin mitchell neil norman oscar phillip randall
ronnie shane stanley tommy trevor victor warren wesley alex hector omar rafael ruben tyrone
""")

ROLE_LOCALS = _wordset("""
sales quotes quote rfq rfqs purchasing purchase info orders order accounting ap ar admin office
support service noreply no-reply donotreply do-not-reply estimating estimates engineering quality
shipping receiving buyer buyers procurement contact hello mail team inquiries inquiry marketing
billing invoices invoice payables receivables help helpdesk it hr careers jobs webmaster postmaster
customerservice cs operations ops production planning scheduling mfg manufacturing reception
front frontdesk general contracts supplier suppliers vendors vendor docs documents drawings
""")

PUBLIC_MAIL_DOMAINS = _wordset("""
gmail.com googlemail.com yahoo.com ymail.com outlook.com hotmail.com live.com msn.com icloud.com
me.com mac.com aol.com comcast.net att.net sbcglobal.net verizon.net protonmail.com proton.me
gmx.com gmx.net mail.com zoho.com cox.net charter.net earthlink.net bellsouth.net frontier.com
windstream.net centurylink.net optonline.net rocketmail.com yahoo.co.uk hotmail.co.uk
""")

# Web services whose host names identify no one. Their host is kept (without the tenant part);
# the path is replaced, because share links and safe-link wrappers carry names and addresses.
PUBLIC_SERVICE_DOMAINS = _wordset("""
wetransfer.com we.tl dropbox.com box.com sharepoint.com onedrive.com live.com 1drv.ms google.com
goo.gl linkedin.com outlook.com office.com office365.com microsoft.com aka.ms zoom.us youtube.com
youtu.be facebook.com twitter.com x.com instagram.com urldefense.com urldefense.proofpoint.com
proofpoint.com mimecast.com safelinks.protection.outlook.com github.com hightail.com egnyte.com
sharefile.com adobe.com docusign.net docusign.com ups.com fedex.com usps.com dhl.com mcmaster.com
grainger.com apple.com teams.microsoft.com bit.ly tinyurl.com
""")

# Sub-domain labels that carry no identity; others ("acme-portal") are dropped.
GENERIC_LABELS = _wordset("www www2 mail email smtp portal secure app apps shop store files ftp share "
                          "support info go my web cloud vpn remote en us")

# Prefixes of specs and document numbers that look like part numbers but are not.
SPEC_PREFIXES = _wordset("""
mil ams as asme ansi astm iso sae qq nas ms an din jis en ipc aws itar ear eccn cage dfars far
nadcap uns aisi unc unf npt nptf ra rms hrc iatf rohs reach ppap fai cmm cnc edm po rfq rfp rfi
so wo ncr car scar ecn eco ecr rma inv ref job tkt sow usml cui bs ul csa ce fcc
""")

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia", "PR": "Puerto Rico",
}
PROVINCES = "AB BC MB NB NL NS NT NU ON PE QC SK YT".split()

NAME_PARTICLES = _wordset("van von der den de la le du di da del dos das st st. mac")
HONORIFICS = _wordset("mr mr. mrs mrs. ms ms. miss mx mx. dr dr. prof prof.")
NAME_SUFFIXES = _wordset("jr jr. sr sr. ii iii iv pe p.e. phd ph.d. mba cpim cscp cqe cpsm pmp "
                         "cmfge csp esq esq. md cpa")
GREETING_SKIP = _wordset("team all everyone everybody there folks guys gang sir sirs madam "
                         "gentlemen ladies friends colleagues again both you y'all yall")
MONTHS_DAYS = _wordset("january february march april may june july august september october "
                       "november december jan feb mar apr jun jul aug sep sept oct nov dec monday "
                       "tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur "
                       "thurs fri sat sun")

ROLE_WORDS = ROLE_LOCALS | _wordset("desk team dept department notifications notification alerts mailbox "
                                     "inbox reply system bot automated rfqs")
NEVER_NAMES = _wordset("am pm a.m p.m re fw fwd cc bcc attn ext tel fax")
NOT_A_NAME = COMMON_WORDS | INDUSTRY_WORDS | TITLE_WORDS | MONTHS_DAYS | _wordset(
    " ".join(STATES.values())) | {s.lower() for s in STATES} - {"al", "ed"}
AMBIGUOUS = NOT_A_NAME | NAME_WORDS

# ---------------------------------------------------------------------------------------------
# Patterns

EMAIL_RE = re.compile(r"(?<![\w.+'%-])[A-Za-z0-9](?:[A-Za-z0-9._%+'-]{0,63})@"
                      r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}(?![\w-])")
MAILTO_QUERY_RE = re.compile(r"(?i)(mailto:[^\s?<>\"]+)\?[^\s<>\"\]\)]*")
URL_RE = re.compile(r"(?i)\b(?:(?:https?|ftp)://|www\.)[^\s<>\"'\]\[)(]+")
# Top-level domains for addresses written without http:// or www. Words that end sentences or
# abbreviations ("Dwg.No", "Mfg.Co", "at", "is") are left out, and a domain must be lowercase
# (or all capitals in a capitals signature), so "Acme.Co" in a company name is not a domain.
TLDS = ("com net org edu gov mil us co io biz info ca mx uk de fr nl se ch au jp cn eu tech ai app "
        "dev aero online site store industries email cloud ly gl tl es pl cz dk fi ie nz sg kr tw br "
        "ar cl il za ru")
BARE_DOMAIN_RE = re.compile(
    r"(?<![\w@./-])((?:[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\.)+(?:" + "|".join(TLDS.split()) + r")"
    r"|(?:[A-Z0-9](?:[A-Z0-9-]{0,62}[A-Z0-9])?\.)+(?:COM|NET|ORG|US|EDU|GOV))(?![\w-]|\.[A-Za-z0-9])")

_EXT = r"(?P<ext>[ \t]*(?:,[ \t]*)?(?:x|ext\.?|extension|ext:)[ \t]*\.?[ \t]*\d{1,6})?"
PHONE_NANP_RE = re.compile(
    r"(?<![\w+-])(?P<num>(?:\+?1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4})" + _EXT
    + r"(?![\d]|-\d)", re.I)
PHONE_INTL_RE = re.compile(
    r"(?<![\w+])(?P<num>\+(?!1[ .(-]?\d{3}[ .)-])[1-9]\d{0,2}(?:[ .-]?(?:\(0\)|\(\d{1,4}\)|\d{1,5})){2,7})"
    + _EXT + r"(?![\d])", re.I)
PHONE_LABEL_RE = re.compile(
    r"(?i)(?<![\w])(?P<label>tel(?:ephone)?|phone|ph|mobile|mob|cell(?:ular)?|fax|facsimile|"
    r"direct(?:[ \t]+line)?|dir|office|off|main|work|toll[- ]free|whats[ ]?app|[pmcfodtwh](?=[ \t]*[:.]))"
    r"[ \t]*[:.#]?[ \t]*(?:\([a-z]+\)[ \t]*)?(?P<num>(?:\+|00)?[\d(][\d \t().-]{5,22}\d)" + _EXT
    + r"(?![\d])")

_STREET_SUF = (r"(?:Street|St|Avenue|Ave|Av|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|"
               r"Circle|Cir|Parkway|Pkwy|Pky|Place|Pl|Highway|Hwy|Terrace|Ter|Trail|Trl|Loop|Pike|"
               r"Plaza|Plz|Crossing|Xing|Expressway|Expy|Freeway|Fwy|Turnpike|Tpke|Square|Sq|"
               r"Crescent|Cres)")
_DIR = r"(?:N|S|E|W|NE|NW|SE|SW|North|South|East|West)"
_SWORD = r"(?:[A-Z][A-Za-z'\-]*\.?|\d{1,3}(?:st|nd|rd|th|ST|ND|RD|TH))"
_UNIT = (r"(?:Suite|Ste\.?|Unit|Bldg\.?|Building|Fl\.?|Floor|Rm\.?|Room|Apt\.?|Mail[ \t]?Stop|"
         r"M/S|#)[ \t]*#?[ \t]*[A-Z0-9][A-Z0-9\-]{0,5}")
STREET_RE = re.compile(
    r"(?<![\w#$.\-/])(?P<num>\d{1,6}(?:-?[A-Z](?![a-z]))?(?:-\d{1,5})?)[ \t]+"
    r"(?P<name>(?:" + _DIR + r"\.?[ \t]+)?" + _SWORD + r"(?:[ \t]+" + _SWORD + r"){0,3}?)"
    r"[ \t]+(?P<suf>(?i:" + _STREET_SUF + r"))\b\.?"
    r"(?P<post>[ \t]+" + _DIR + r"\b\.?)?"
    r"(?P<unit>(?:,[ \t]*|[ \t]+)" + _UNIT + r"\b)?")
STREET_WORDS = _wordset("main north south east west n s e w state center market high park church mill "
                        "lake river spring industrial commerce enterprise technology business corporate "
                        "airport railroad depot factory")
LOOSE_SUFFIXES = _wordset("way dr pl ct cir loop pike sq ter")
ROUTE_RE = re.compile(r"(?i)(?<![\w#$.\-/])\d{1,6}[ \t]+(?:(?:US|State|County|Co\.?|FM)[ \t]+)?"
                      r"(?:Highway|Hwy|Route|Rte|Road|Rd)\.?[ \t]+\d{1,4}[A-Z]?\b")
UNIT_RE = re.compile(r"(?i)\b(?P<label>Suite|Ste\.?|Mail[ \t]?Stop)[ \t]*#?[ \t]*(?P<num>\d[\dA-Z\-]{0,5})\b")
POBOX_RE = re.compile(r"(?i)\b(?P<label>(?:P\.?[ \t]*O\.?|Post[ \t]+Office)[ \t]*Box)[ \t]+(?P<num>\d{1,7})\b")
_STATE_ALT = "|".join(sorted(STATES, key=len, reverse=True))
_STATE_FULL_ALT = "|".join(sorted((re.escape(v) for v in STATES.values()), key=len, reverse=True))
_CITY = r"(?P<city>[A-Z][A-Za-z.'\-]*(?:[ \t]+[A-Z][A-Za-z.'\-]*){0,3})"
CITY_ZIP_RE = re.compile(_CITY + r",?[ \t]+(?P<state>(?:" + _STATE_ALT + r")\b|(?:" + _STATE_FULL_ALT
                         + r")\b)\.?,?[ \t]+(?P<zip>\d{5}(?:-\d{4})?)(?![\d-])")
CITY_CA_RE = re.compile(_CITY + r",?[ \t]+(?P<state>(?:" + "|".join(PROVINCES) + r"))\b\.?,?[ \t]+"
                        r"(?P<zip>[A-Z]\d[A-Z][ \t]?\d[A-Z]\d)\b")
CITY_ST_RE = re.compile(_CITY + r",[ \t]*(?P<state>(?:" + _STATE_ALT + r")\b|(?:" + _STATE_FULL_ALT
                        + r")\b)(?![\w-])(?!\.\w)")

COMPANY_SUFFIX_RE = re.compile(
    r"(?<![\w&])(?P<suf>Inc\b\.?|Incorporated\b|LLC\b|L\.L\.C\.|LLP\b|Corp\b\.?|Corporation\b|"
    r"Co\.(?!\w)|Company\b|Ltd\b\.?|Limited\b|GmbH\b|AG\b|S\.A\.|PLC\b|Pty\.?[ \t]+Ltd\b\.?|"
    r"B\.V\.|S\.p\.A\.|S\.r\.l\.|ULC\b|Ltda\b\.?|SARL\b|S\.A\.S\.|S\.L\.(?!\w)|Pte\.?[ \t]+Ltd\b\.?|"
    r"Sdn\.?[ \t]+Bhd\b\.?|K\.K\.|A/S\b|ApS\b|S\.[ \t]?de[ \t]R\.L\.)", re.I)
_SUFFIX_WORDS = _wordset("inc inc. incorporated llc l.l.c. llp corp corp. corporation co co. company "
                         "ltd ltd. limited gmbh ag s.a. plc pty b.v. s.p.a. s.r.l. ulc ltda ltda. sarl "
                         "s.a.s. s.l. pte sdn bhd k.k. a/s aps")

SIGNOFF_RE = re.compile(
    r"^(?P<phrase>(?:many\s+)?thanks?(?:\s+(?:you|again|so\s+much|very\s+much|in\s+advance|"
    r"for\s+your\s+help|for\s+your\s+time))*|thank\s+you(?:\s+(?:again|so\s+much|very\s+much|"
    r"in\s+advance))?|thx|tia|regards|best(?:\s+(?:regards|wishes))?|kind(?:est)?\s+regards|"
    r"warm(?:est)?\s+regards|with\s+(?:best\s+)?regards|sincerely(?:\s+yours)?|yours\s+truly|"
    r"cheers|respectfully|cordially|all\s+the\s+best|take\s+care|talk\s+soon|v/r|vr|"
    r"very\s+respectfully|appreciate\s+it|much\s+appreciated|--)"
    r"(?P<punct>[\s,.!;:\-]*)(?P<rest>.*)$", re.I)
REPLY_BOUNDARY_RE = re.compile(
    r"^(?:[>*\s]*(?:From|Sent|To|Cc|Subject|Date)\s*:\*?\s|-{2,}\s*(?:Original|Forwarded)\s+Message|"
    r"_{8,}|On\s.{4,}\bwrote:?\s*$|Begin forwarded message|From:\s)", re.I)
QHDR_RE = re.compile(r"^[ \t>*]*(?P<k>From|To|Cc|Bcc|Reply-To|Sender|Sent by)[ \t]*\*?:\*?[ \t]*(?P<v>.+)$",
                     re.I | re.M)
WROTE_RE = re.compile(r"(?:^|\n)[ \t>]*On[ \t][^\n]{4,160}?(?:\n[ \t>]*[^\n]{0,160}?)?\bwrote:", re.I)
GREETING_RE = re.compile(
    r"^[ \t>]*(?:hi|hello|hey|dear|good[ \t]+(?:morning|afternoon|evening|day)|greetings|morning|"
    r"afternoon|attn:?|attention:?)[ \t,]+(?P<names>[^\n,:;!?]{1,60})", re.I | re.M)
HONORIFIC_RE = re.compile(r"(?<![\w])(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?[ \t]+(?P<a>[A-Z][A-Za-z'\-]+)"
                          r"(?:[ \t]+(?P<b>[A-Z][A-Za-z'\-]+))?")
PROSE_NAME_RE = re.compile(r"(?<![\w])(?P<first>[A-Z][a-z]+)[ \t]+(?:(?P<mi>[A-Z])\.?[ \t]+)?"
                           r"(?P<last>(?:Mc|Mac|O')?[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)(?![\w])")
_PAIR = (r"(?P<first>[A-Z][a-z]+(?:-[A-Z][a-z]+)?)[ \t]+(?:(?P<mi>[A-Z])\.?[ \t]+)?"
         r"(?P<last>(?:Mc|Mac|O')?[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)(?![\w])")
CONTEXT_BEFORE_RE = re.compile(
    r"(?<![\w])(?i:cc|cc'd|copy|copying|attn|attention|contact|contacting|ask|asked|call|called|email|"
    r"e-mail|emailed|ping|loop[ \t]+in|looping[ \t]+in|reach[ \t]+out[ \t]+to|reach|talk[ \t]+to|talked[ \t]+to|"
    r"speak[ \t]+with|spoke[ \t]+with|spoke[ \t]+to|per|thanks[ \t]+to|according[ \t]+to|approved[ \t]+by|"
    r"requested[ \t]+by|signed[ \t]+by|sent[ \t]+by|forwarded[ \t]+by)[ \t]*:?[ \t]+" + _PAIR)
CONTEXT_AFTER_RE = re.compile(
    r"(?<![\w])" + _PAIR + r"[ \t]+(?:will|would|said|says|mentioned|asked|wants|called|emailed|sent|wrote|"
    r"told|suggested|requested|confirmed|approved|noted|handles|owns|runs|manages|is[ \t]+out|"
    r"is[ \t]+our|from[ \t]+our|at[ \t]+our|in[ \t]+our|of[ \t]+our)\b")
LABELED_COMPANY_RE = re.compile(
    r"(?im)^[ \t>*]*(?:company|customer|vendor|supplier|organi[sz]ation|firm|sold[ -]to|ship[ -]to|"
    r"bill[ -]to|end[ -]user)(?:[ \t]+name)?[ \t]*:[ \t]*(?P<v>[A-Z][^\n,;|\d]{1,60}?)[ \t]*(?:[,;|\n]|$)")
PART_LABEL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:P/?N|Part(?:[ \t]*(?:No\.?|Number|Num\.?|#))?|Dwg\.?(?:[ \t]*(?:No\.?|#))?|"
    r"Drawing(?:[ \t]*(?:No\.?|Number|#))?|Item[ \t]*(?:No\.?|#)|Model[ \t]*(?:No\.?|#)|CPN|MPN)"
    r"[ \t]*[:#.]?[ \t]*(?P<pn>[A-Z0-9][A-Z0-9./_\-]*[A-Z0-9])")
PART_SHAPE_RE = re.compile(r"(?<![A-Za-z0-9\-])(?:[A-Z]{1,5}-?\d{3,}[A-Z]?(?:[-.][A-Z0-9]{1,6})*|"
                           r"\d{3,}-\d{2,}(?:-[A-Z0-9]{1,6})*)(?![A-Za-z0-9])")
REFNUM_BEFORE_RE = re.compile(r"(?i)(?:\b(?:po|p\.o\.|rfq|rfp|quote|qte|order|invoice|inv|so|job|"
                              r"ref|case|ticket|tracking|ecn|eco|ncr|rma|lot|serial|s/n)\b[\s#:.-]*)$")

# Text replaced so far is held as one private-use character per value, so later patterns cannot
# match inside a fake (or a fake can never be scrubbed a second time).
_PH_BASE = 0xF0000
_PH_RE = re.compile("[\U000F0000-\U000FFFFD]")
_B = r"(?<![^\W_])"   # token boundaries that treat "_" as a separator (file names)
_E = r"(?![^\W_])"


# ---------------------------------------------------------------------------------------------
# Small helpers

# The repo allows no em or en dashes (tests/test_import.py checks every fixture), and Outlook
# turns "--" into one as you type, so every dash-like character becomes a plain hyphen.
DASHES_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]")


def plain_dashes(text: str) -> Tuple[str, int]:
    return DASHES_RE.subn("-", text or "")


def _esc(word: str) -> str:
    """re.escape, with a straight apostrophe also matching a curly one (O'Brien, O\u2019Brien)."""
    return re.escape(word).replace("'", "['\u2019]")


def _h(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()[:8], "big")


def _hex(*parts: str, n: int = 8) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:n]


def fold(text: str) -> str:
    """ASCII form of a name: accents dropped (Jose for Jos\u00e9)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", fold(text).lower())


def match_case(src: str, fake: str) -> str:
    letters = [c for c in src if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return fake.upper()
    if letters and all(c.islower() for c in letters):
        return fake.lower()
    return fake


def is_titleish(word: str) -> bool:
    return bool(word) and word[0].isupper()


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def registrable(host: str) -> Tuple[List[str], List[str]]:
    """Split a host into (sub-domain labels, registrable labels): mail.acme.co.uk ->
    (["mail"], ["acme", "co", "uk"])."""
    labels = [x for x in host.lower().strip(".").split(".") if x]
    n = 2
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in ("co", "com", "net", "org", "ac", "gov", "ltd", "plc"):
        n = 3
    return labels[:-n], labels[-n:]


def _host_in(host: str, domains: Iterable[str]) -> bool:
    host = host.lower().strip(".")
    return any(host == d or host.endswith("." + d) for d in domains)


def is_reserved_domain(host: str) -> bool:
    host = host.lower().strip(".")
    return (_host_in(host, ("example.com", "example.org", "example.net", "localhost"))
            or host.endswith((".example", ".test", ".invalid", ".localhost")) or host in ("example", "test"))


def is_public_mail(host: str) -> bool:
    return _host_in(host, PUBLIC_MAIL_DOMAINS)


def is_public_service(host: str) -> bool:
    return _host_in(host, PUBLIC_SERVICE_DOMAINS) or is_public_mail(host)


def split_recipient(value: str) -> Tuple[str, str]:
    """'Name <addr>', 'Name [mailto:addr]', 'addr', or 'Name' -> (name, addr)."""
    value = (value or "").strip()
    m = re.match(r'^\s*"?(?P<n>.*?)"?\s*[<\[(]\s*(?:mailto:)?(?P<a>[^<>\[\]()\s]+@[^<>\[\]()\s]+)\s*[>\])]\s*$',
                 value, re.I)
    if m:
        return m.group("n").strip().strip('"').strip(), m.group("a").strip()
    m = EMAIL_RE.fullmatch(value.strip("<>").strip())
    if m:
        return "", m.group(0)
    return value.strip('"').strip(), ""


def split_recipient_list(value: str) -> List[str]:
    value = value.strip()
    if ";" in value:
        parts = value.split(";")
    elif value.count("<") > 1 or value.lower().count("mailto:") > 1:
        parts = re.split(r"(?<=[>\]\)])\s*,\s*", value)
    elif "," in value and "@" not in value:
        bits = [b.strip() for b in value.split(",") if b.strip()]
        # "Doe, Jane" is one person written last name first; "Jane Doe, Bob Ruiz" is two.
        if len(bits) == 2 and all(len(b.split()) == 1 for b in bits):
            parts = [value]
        else:
            parts = bits
    else:
        parts = [value]
    return [p.strip() for p in parts if p.strip()]


def strip_quote(line: str) -> str:
    return re.sub(r"^[ \t]*(?:>[ \t]?)+", "", line)


def looks_like_name_token(tok: str) -> bool:
    core = tok.replace("-", "").replace("'", "").replace("\u2019", "").replace(".", "")
    return bool(core) and core.isalpha() and (tok[0].isupper() or tok.isupper())


# ---------------------------------------------------------------------------------------------
# Name parsing

class ParsedName:
    __slots__ = ("first", "middle", "last", "company", "loose")

    def __init__(self) -> None:
        self.loose = False
        self.first = ""
        self.middle: List[str] = []
        self.last = ""
        self.company: List[str] = []

    @property
    def is_person(self) -> bool:
        return bool(self.first or self.last)


def parse_display_name(display: str, loose: bool = False) -> ParsedName:
    """A display name or signature line -> person parts and company candidates.

    loose: the text sits where a name is expected (after a sign-off, in a From header), so a
    name that is also a common word ("Mark Price") still counts.
    """
    out = ParsedName()
    out.loose = loose
    text = _norm_ws(display.replace("\u2019", "'")).strip("\"' ")
    if not text or "@" in text:
        return out
    # "Jane Doe (Acme)", "Jane Doe | Acme Precision", "Jane Doe - Acme"
    for inner in re.findall(r"[(\[]([^)\]]+)[)\]]", text):
        out.company.append(inner.strip())
    text = re.sub(r"[(\[][^)\]]*[)\]]", " ", text)
    pieces = [p.strip() for p in re.split(r"\s+[|\u2022\u00b7/]\s+|\s+-\s+|\s*\|\s*", text) if p.strip()]
    if not pieces:
        return out
    text, extra = pieces[0], pieces[1:]
    out.company.extend(extra)
    # "Doe, Jane Q." and "Jane Doe, PE"
    if "," in text:
        left, _, right = text.partition(",")
        ltoks, rtoks = left.split(), right.split()
        if rtoks and all(t.lower().strip(".") in NAME_SUFFIXES or t.lower() in NAME_SUFFIXES for t in rtoks):
            text = left
        elif ltoks and all(t.lower().strip(".") in NEVER_NAMES or t.lower() in NOT_A_NAME for t in ltoks):
            text = right
        elif 1 <= len(ltoks) <= 3 and (len(rtoks) == 1 or len(rtoks) == 2 and len(rtoks[1].rstrip(".")) == 1):
            text = right.strip() + " " + left.strip()
        else:
            return out
    toks = [t for t in text.split() if t.lower() not in HONORIFICS and t.lower().strip(",") not in NAME_SUFFIXES]
    toks = [t.strip(",") for t in toks if t.strip(",")]
    if not toks or len(toks) > 5:
        return out
    if not all(looks_like_name_token(t) or t.lower() in NAME_PARTICLES for t in toks):
        return out
    words = [t for t in toks if len(t.rstrip(".")) > 1 or not t.isalpha()]
    lowered = [t.lower().strip(".") for t in toks]
    blocked = AMBIGUOUS
    # A display name like "Acme Precision" or "Acme Quotes" is a company or a mailbox.
    if any(w in INDUSTRY_WORDS or w in TITLE_WORDS or w in _SUFFIX_WORDS or w in ROLE_WORDS
           for w in lowered if w not in NAME_PARTICLES):
        rest = [t for t in toks if t.lower().strip(".") not in ROLE_WORDS]
        if rest:
            out.company.append(" ".join(rest))
        return out
    if not words:
        return out
    name_toks = [t for t in toks]
    # Initials in the middle ("Jane Q. Doe") are middle initials.
    first = name_toks[0]
    if len(first.rstrip(".")) == 1:
        return out
    rest = name_toks[1:]
    last_parts: List[str] = []
    middle: List[str] = []
    if rest:
        # Particles belong to the surname: "Anna van der Berg".
        i = len(rest) - 1
        last_parts = [rest[i]]
        while i - 1 >= 0 and rest[i - 1].lower() in NAME_PARTICLES:
            i -= 1
            last_parts.insert(0, rest[i])
        middle = rest[:i]
    last = " ".join(last_parts)
    first_l = first.lower().strip(".")
    last_l = last_parts[-1].lower().strip(".") if last_parts else ""
    # In running text neither word may be an ordinary word. Where a name is expected (a From
    # header, a signature) the surname may be one ("Mark Price", "Ilsabet Frame").
    if first_l in NOT_A_NAME or last_l in MONTHS_DAYS or last_l in NEVER_NAMES:
        return out
    if not loose and (first_l in blocked or last_l in blocked):
        return out
    if len(last.rstrip(".")) == 1:     # "Jane D." -> first name only
        last = ""
    out.first = first
    out.middle = [m for m in middle]
    out.last = last
    return out


def local_part_name(local: str) -> Optional[Tuple[str, str]]:
    """jane.doe, jane_doe, jane-doe -> ("Jane", "Doe"). Anything else -> None."""
    base = re.sub(r"\d+$", "", local.lower())
    m = re.fullmatch(r"([a-z]{2,20})[._-]([a-z]{2,24})", base)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if a in ROLE_LOCALS or b in ROLE_LOCALS or a in NOT_A_NAME or b in NOT_A_NAME:
        return None
    return a.capitalize(), b.capitalize()


def local_matches(base: str, first: str, last: str) -> bool:
    """True when a mailbox name is built from this first and last name (jdoe, jane.doe)."""
    f, s = squash(first), squash(last)
    if not f or not s:
        return False
    options = {f, s} if min(len(f), len(s)) >= 4 else set()
    hy = re.sub(r"[^a-z0-9-]", "", fold(last).lower())
    options |= {f[0] + hy, f + "." + hy, hy}
    for sep in (".", "_", "-", ""):
        options |= {f + sep + s, s + sep + f, f[0] + sep + s, f + sep + s[0], s + sep + f[0]}
    return base in options


def mimic_local(local: str, first: str, last: str, ffirst: str, flast: str) -> str:
    """Build a fake mailbox name in the same pattern as the real one (jdoe -> nfarrow)."""
    lo = local.lower()
    digits = re.search(r"\d+$", lo)
    tail = digits.group(0) if digits else ""
    base = lo[: len(lo) - len(tail)] if tail else lo
    f, s = squash(first), squash(last)
    ff, fs = squash(ffirst) or "alex", squash(flast) or "doe"
    if tail:
        tail = str(_h("tail", lo) % (10 ** len(tail))).zfill(len(tail))
    options: List[Tuple[str, str]] = []
    hy = re.sub(r"[^a-z0-9-]", "", fold(last).lower())
    if f and hy and hy != s:        # pashworth-lund for Priscilla Ashworth-Lund
        options += [(f[0] + hy, ff[0] + fs), (f + "." + hy, ff + "." + fs), (hy, fs)]
    if f and s:
        for sep in (".", "_", "-", ""):
            options += [(f + sep + s, ff + sep + fs), (s + sep + f, fs + sep + ff),
                        (f[0] + sep + s, ff[0] + sep + fs), (f + sep + s[0], ff + sep + fs[0]),
                        (s + sep + f[0], fs + sep + ff[0])]
    if f:
        options.append((f, ff))
    if s:
        options.append((s, fs))
    for real, fake in options:
        if base == real:
            return fake + tail
    return (ff[0] + fs if f or s else ff + "." + fs) + tail


# ---------------------------------------------------------------------------------------------
# The persistent map

class ScrubMap:
    """Real value -> fake value, per kind, kept in private_emails/.scrub_map.json."""

    SECTIONS = ("first", "last", "people", "email", "domain", "company", "company_forms",
                "company_weak", "phone", "street", "city", "part", "manual")

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.data: Dict[str, Any] = {
            "about": "Real-to-fake map for tools/scrub_email.py. It holds real names: never commit it.",
            "version": 1,
        }
        for s in self.SECTIONS:
            self.data[s] = {}
        if path is not None and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SystemExit(f"Cannot read the scrub map {path}: {exc}")
            for s in self.SECTIONS:
                if isinstance(loaded.get(s), dict):
                    self.data[s].update(loaded[s])
        # Kept up to date as values are added, so picking a fake stays fast with a big map.
        self._used: Dict[str, Set[str]] = {s: set() for s in self.SECTIONS}
        self._reals: Set[str] = set()
        for s in self.SECTIONS:
            for k, v in self.data[s].items():
                self._note(s, k, v)

    def _note(self, section: str, key: str, value: Any) -> None:
        if isinstance(value, str):
            self._used[section].add(value.lower())
        if section in ("first", "last"):
            self._reals.add(key.lower())
        elif section == "company":
            self._reals.update(key.lower().replace("whole:", "").split())
        elif section == "city":
            self._reals.update(key.split("|")[0].lower().split())

    def get(self, section: str, key: str) -> Any:
        return self.data[section].get(key)

    def set(self, section: str, key: str, value: Any) -> None:
        self.data[section][key] = value
        self._note(section, key, value)

    def used(self, section: str) -> Set[str]:
        return self._used[section]

    def real_tokens(self) -> Set[str]:
        """Every real name word the map knows, so no fake is ever also a real name."""
        return self._reals

    def pick(self, section: str, key: str, candidates: Sequence[str]) -> str:
        """The fake for key: the stored one, else a candidate chosen from a hash of the key
        (so an empty map still gives mostly the same answers), skipping fakes already used and
        real names."""
        got = self.data[section].get(key)
        if isinstance(got, str):
            return got
        used = self._used[section]
        reals = self._reals
        n = len(candidates)
        start = _h(section, key) % n
        choice = candidates[start]
        for i in range(n):
            c = candidates[(start + i) % n]
            if c.lower() not in used and not (set(c.lower().split()) & reals):
                choice = c
                break
        self.set(section, key, choice)
        return choice

    def fakes(self) -> Set[str]:
        """Every fake word in the map (lowercase), for the "check these" scan."""
        out: Set[str] = set()
        for s in ("first", "last", "company", "street", "city"):
            for v in self.data[s].values():
                if isinstance(v, str):
                    out.update(w.lower() for w in re.split(r"[\s|]+", v) if w)
        return out

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------------------------
# The scrubber

class _Slots:
    def __init__(self) -> None:
        self.values: List[str] = []

    def put(self, text: str) -> str:
        self.values.append(text)
        return chr(_PH_BASE + len(self.values) - 1)

    def restore(self, text: str) -> str:
        return _PH_RE.sub(lambda m: self.values[ord(m.group(0)) - _PH_BASE], text)


class Scrubber:
    def __init__(self, smap: ScrubMap, manual: Sequence[Tuple[str, str]] = (),
                 part_numbers: bool = False) -> None:
        self.map = smap
        self.part_numbers = part_numbers
        for real, fake in manual:
            self.map.set("manual", real.strip().lower(), fake.strip())
        self.email_person: Dict[str, str] = {}      # address -> person key, for this run
        self.emails_seen: Set[str] = set()
        self.domain_stems: Set[str] = set()
        self.parts: Counter = Counter()             # part numbers seen in this run
        self.part_labeled: Set[str] = set()
        self.stats: Counter = Counter()             # (kind, real, fake) -> count
        self.leak_terms: Set[Tuple[str, str]] = set()   # (kind, real text) replaced in this run
        self._cache: Dict[str, Any] = {}
        self._texts: List[str] = []
        self._sender_first = ""
        for real, fake in self.manual_pairs():
            self._seed_person_from_manual(real, fake)

    # -- registry ------------------------------------------------------------------------------

    def manual_pairs(self) -> List[Tuple[str, str]]:
        return sorted(self.map.data["manual"].items(), key=lambda kv: -len(kv[0]))

    def _dirty(self) -> None:
        self._cache.clear()

    def _seed_person_from_manual(self, real: str, fake: str) -> None:
        """--map "Robert Ruiz=Tom Becker" also maps "Robert" alone to "Tom"."""
        if "@" in real or "." in real.replace(". ", ""):
            return
        pr = parse_display_name(real.title(), loose=True)
        pf = parse_display_name(fake, loose=True)
        if pr.first and pr.last and pf.first and pf.last:
            self.map.set("first", pr.first.lower(), pf.first)
            self.map.set("last", pr.last.lower(), pf.last)
            self.add_person(pr.first, pr.last, pr.middle, allow_common=True)

    def add_name_token(self, section: str, token: str, allow_common: bool = False) -> None:
        """allow_common: the token sits where a name is certain (a From header), so an ordinary
        word ("Price") is registered too; alone it is then replaced only in name contexts."""
        key = token.lower().strip(".")
        if len(key) < 2 or key in NAME_PARTICLES or (key in NOT_A_NAME and not allow_common):
            return
        fake = self.map.pick(section, key, FAKE_FIRST if section == "first" else FAKE_LAST)
        folded = fold(key)
        if folded != key and folded and not self.map.get(section, folded):
            self.map.set(section, folded, fake)
        if section == "last" and ("-" in key or " " in key):
            parts = [p for p in re.split(r"[-\s]+", key) if len(p) >= 3 and p not in NAME_PARTICLES and p not in NOT_A_NAME]
            for part in parts:
                if len(re.split(r"\s+", key)) > 1 and len([w for w in key.split() if w not in NAME_PARTICLES]) == 1:
                    # "dos Santos": "Santos" alone is the same surname.
                    if not self.map.get("last", part):
                        self.map.set("last", part, fake)
                else:
                    self.map.pick("last", part, FAKE_LAST)
        self._dirty()

    def add_person(self, first: str, last: str, middle: Sequence[str] = (), allow_common: bool = False) -> str:
        first, last = first.strip(), last.strip()
        if first:
            self.add_name_token("first", first)
        if last:
            self.add_name_token("last", last, allow_common)
        for m in middle:
            if len(m.rstrip(".")) > 1:
                self.add_name_token("first", m)
        key = f"{first.lower()}|{last.lower()}"
        if first and last:
            rec = self.map.get("people", key) or {"first": first, "last": last, "middle": []}
            for m in middle:
                if m not in rec["middle"]:
                    rec["middle"].append(m)
            self.map.set("people", key, rec)
        self._dirty()
        return key

    def add_parsed(self, parsed: ParsedName, addr: str = "") -> None:
        if parsed.is_person:
            key = self.add_person(parsed.first, parsed.last, parsed.middle, allow_common=parsed.loose)
            if addr:
                self.email_person[addr.lower()] = key
        for c in parsed.company:
            self.add_company(c)

    def add_company(self, name: str) -> None:
        """Register a company name. Distinctive words get a fake stem; a name of generic words
        only gets a whole fake name."""
        words = company_core(name)
        if not words:
            return
        lowered = [w.lower() for w in words]
        runs = distinctive_runs(words)
        if runs:
            for run in runs:
                key = " ".join(w.lower() for w in run)
                if len(squash(key)) < 3:
                    continue
                self.map.pick("company", key, COMPANY_STEMS)
                # "Acme-Torvald" is also written "Torvald".
                for w in run:
                    for part in w.split("-") if "-" in w else ():
                        if len(part) >= 4 and is_distinctive(part) and part.lower() not in AMBIGUOUS:
                            self.map.pick("company", part.lower(), COMPANY_STEMS)
        else:
            if len(words) < 2 and not any(w.isupper() and len(w) > 2 for w in words):
                return          # one generic word ("Precision") is no company name
            key = " ".join(lowered)
            stem = self.map.pick("company", "whole:" + key, COMPANY_STEMS)
            trade = COMPANY_TRADES[_h("trade", key) % len(COMPANY_TRADES)][0]
            if not self.map.get("company", key):
                self.map.set("company", key, f"{stem} {trade}")
        self.map.set("company_forms", " ".join(lowered), True)
        self._dirty()

    def note_domain(self, host: str) -> None:
        host = host.lower().strip(".")
        if not host or is_reserved_domain(host) or is_public_service(host):
            return
        _, reg = registrable(host)
        if reg:
            self.domain_stems.add(reg[0])

    def link_domains(self) -> None:
        """Give every domain stem a company: the company or company word that spells the stem
        ("Realshop" for realshop.com), else the stem's own distinctive part ("acme" of
        acmeprecision.com)."""
        forms = {squash(k): k for k in self.map.data["company_forms"]}
        runs = {squash(k): k for k in self.map.data["company"] if not k.startswith("whole:")}
        for stem in sorted(self.domain_stems):
            if self.map.get("domain", stem + "|stem"):
                continue
            s = squash(stem)
            link = forms.get(s) or runs.get(s)
            if not link:
                words = [w for w in re.split(r"[-_]+", stem) if w]
                if len(words) == 1:
                    words = split_stem(stem)
                if squash(words[0]) in runs:
                    link = " ".join([runs[squash(words[0])]] + words[1:])
                else:
                    # Seen only in the domain, so a guess at the word boundary: replaced in the
                    # text only when capitalized ("Real" of realshop.com, never "real").
                    for run in distinctive_runs(words):
                        self.map.set("company_weak", " ".join(w.lower() for w in run), True)
                    self.add_company(" ".join(w.capitalize() for w in words))
                    link = " ".join(w.lower() for w in words)
            self.map.set("domain", stem + "|stem", link)
        self._dirty()

    # -- fakes ---------------------------------------------------------------------------------

    def fake_company_words(self, words: Sequence[str]) -> List[str]:
        """Real company words -> fake words, generic words kept."""
        lowered = [w.lower() for w in words]
        whole = self.map.get("company", " ".join(lowered))
        if whole and not distinctive_runs(words):
            return whole.split()
        out: List[str] = []
        i = 0
        while i < len(words):
            if is_distinctive(words[i]):
                j = i
                while j < len(words) and is_distinctive(words[j]):
                    j += 1
                key = " ".join(lowered[i:j])
                fake = self.map.get("company", key) or self.map.pick("company", key, COMPANY_STEMS)
                out.append(fake)
                i = j
            else:
                out.append(words[i])
                i += 1
        return out

    def fake_domain(self, host: str) -> str:
        host = host.lower().strip(".")
        manual = self.map.get("manual", host)
        if manual:
            return manual.lower()
        if is_reserved_domain(host):
            return host
        sub, reg = registrable(host)
        if is_public_service(host):
            return ".".join([s for s in sub if s in GENERIC_LABELS] + reg)
        regkey = ".".join(reg)
        manual = self.map.get("manual", regkey)
        fake_reg = manual.lower() if manual else self.map.get("domain", regkey)
        if not fake_reg:
            stem = reg[0]
            self.note_domain(host)
            self.link_domains()
            form = self.map.get("domain", stem + "|stem") or stem
            fake_words = self.fake_company_words(form.split())
            base = squash("".join(fake_words)) or "scrubbed"
            fake_reg = ".".join([base] + reg[1:])
            used = {v for k, v in self.map.data["domain"].items() if not k.endswith("|stem")}
            n = 2
            while fake_reg in used:
                fake_reg = ".".join([f"{base}{n}"] + reg[1:])
                n += 1
            self.map.set("domain", regkey, fake_reg)
        return ".".join([s for s in sub if s in GENERIC_LABELS] + [fake_reg])

    def fake_email(self, addr: str) -> str:
        a = addr.strip().lower()
        manual = self.map.get("manual", a)
        if manual:
            return manual
        got = self.map.get("email", a)
        if got:
            return got
        local, _, host = a.rpartition("@")
        if not local or is_reserved_domain(host):
            return a
        fhost = self.fake_domain(host)
        base = re.sub(r"[\d._+-]+$", "", local)
        person = self.email_person.get(a)
        if base in ROLE_LOCALS and not is_public_mail(host):
            flocal = local
        elif person and person != "|":
            first, _, last = person.partition("|")
            ff = self.fake_token(first, "first") if first else ""
            fl = self.fake_token(last, "last") if last else ""
            flocal = mimic_local(local, first, last, ff or fake_first_for(local), fl or fake_last_for(local))
        else:
            guess = local_part_name(local)
            if guess:
                self.add_person(*guess)
                flocal = mimic_local(local, guess[0], guess[1], self.fake_token(guess[0], "first"),
                                     self.fake_token(guess[1], "last"))
            else:
                ff = FAKE_FIRST[_h("lf", local) % len(FAKE_FIRST)]
                fl = FAKE_LAST[_h("ll", local) % len(FAKE_LAST)]
                flocal = (ff[0] + fl).lower() if len(base) > 2 else ff.lower()
                digits = re.search(r"\d+$", local)
                if digits:
                    flocal += str(_h("d", local) % (10 ** len(digits.group(0)))).zfill(len(digits.group(0)))
        fake = f"{flocal}@{fhost}"
        used = self.map.used("email")
        n = 2
        while fake in used:
            fake = f"{flocal}{n}@{fhost}"
            n += 1
        self.map.set("email", a, fake)
        return fake

    def fake_token(self, token: str, prefer: str = "first") -> str:
        key = token.lower().strip(".")
        order = ("first", "last") if prefer == "first" else ("last", "first")
        for s in order:
            v = self.map.get(s, key) or self.map.get(s, fold(key))
            if v:
                return v
        self.add_name_token(prefer, token)
        return self.map.get(prefer, key) or FAKE_FIRST[_h("t", key) % len(FAKE_FIRST)]

    def fake_phone_digits(self, digits: str, cc: str) -> str:
        """Same number of digits; US numbers become AAA-555-01XX."""
        key = ("+" + digits) if cc and cc != "1" else digits[-10:] if len(digits) >= 10 else digits
        got = self.map.get("phone", key)
        if got and len(got) == len(digits):
            return got
        used = self.map.used("phone")
        for salt in range(200):
            h = _h("phone", key, str(salt))
            if cc == "44" and len(digits) == 12:
                # Ofcom's ranges for drama: 020 7946 0xxx and 07700 900xxx.
                head = "7700900" if digits[2] == "7" else "2079460"
                fake = "44" + head + str(h % 1000).zfill(3)
            elif cc and cc != "1":
                rest_len = len(digits) - len(cc)
                fill = ("55501" + str(h % 10 ** 12).zfill(12))[:rest_len]
                fake = cc + fill
            elif len(digits) >= 10:
                area = FAKE_AREA_CODES[h % len(FAKE_AREA_CODES)]
                fake = digits[:-10] + area + "55501" + str(h // 7 % 100).zfill(2)
            elif len(digits) == 7:
                fake = "55501" + str(h % 100).zfill(2)
            else:
                fake = str(h % 10 ** len(digits)).zfill(len(digits))
            if fake not in used and fake != digits:
                break
        self.map.set("phone", key, fake)
        return fake

    def fake_part(self, part: str) -> str:
        key = part.upper()
        got = self.map.get("part", key)
        if got:
            return got
        used = self.map.used("part")
        for salt in range(50):
            h = hashlib.sha256(f"part\x1f{key}\x1f{salt}".encode()).digest()
            out = []
            for i, ch in enumerate(part):
                b = h[i % len(h)]
                if ch.isdigit():
                    out.append(str(b % 10))
                elif ch.isalpha():
                    c = "ABCDEFGHJKLMNPRSTUVWXYZ"[b % 23]
                    out.append(c if ch.isupper() else c.lower())
                else:
                    out.append(ch)
            fake = "".join(out)
            if fake.upper() != key and fake.lower() not in used:
                break
        self.map.set("part", key, fake)
        return fake

    # -- collecting what to replace -------------------------------------------------------------

    def collect(self, em: Dict[str, Any]) -> None:
        self._collect_address(em.get("from_name") or "", em.get("from_email") or "", sender=True)
        for r in list(em.get("to") or []) + list(em.get("cc") or []):
            name, addr = split_recipient(r)
            self._collect_address(name, addr)
        body = em.get("body") or ""
        subject = em.get("subject") or ""
        names = [a.get("name") or "" for a in em.get("attachments") or []]
        texts = [subject, body] + names
        sender = parse_display_name(em.get("from_name") or "", loose=True)
        self._sender_first = sender.first if sender.is_person else ""
        self._collect_quoted_headers(body)
        self._collect_wrote_lines(body)
        self._collect_greetings(body)
        self._collect_signatures(body)
        for t in texts:
            self._collect_contacts(t)
            self._collect_honorifics(t)
            self._collect_suffix_companies(t)
        for t in (subject, body):
            self._collect_prose_names(t)
        for t in texts:
            self._collect_context_names(t)
            self._collect_parts(t)
            self._collect_cities(t)
        self._texts.extend(texts)
        self._dirty()

    def finish_collect(self) -> None:
        """After every email of the run is collected: company names that spell a domain, then
        a company for every domain."""
        for t in self._texts:
            self._collect_domain_matches(t)
        self.link_domains()
        self._collect_local_links()
        self._texts = []

    def _collect_local_links(self) -> None:
        """A name in the text that spells a mailbox seen in the run ("Anatole Fairleigh" and
        dfairleigh@...) is a person, whatever the first name."""
        pairs: Set[Tuple[str, str]] = set()
        for t in self._texts:
            for m in PROSE_NAME_RE.finditer(t):
                pairs.add((m.group("first"), m.group("last")))
        for addr in sorted(self.emails_seen):
            if addr in self.email_person:
                continue
            base = re.sub(r"\d+$", "", addr.partition("@")[0].lower())
            if base in ROLE_LOCALS or len(base) < 3:
                continue
            for first, last in sorted(pairs):
                if first.lower() in NOT_A_NAME or last.lower() in NOT_A_NAME:
                    continue
                if local_matches(base, first, last):
                    self.email_person[addr] = self.add_person(first, last)
                    break

    def _collect_address(self, name: str, addr: str, sender: bool = False) -> None:
        addr = (addr or "").strip().lower()
        if addr and "@" in addr:
            self.emails_seen.add(addr)
            self.note_domain(addr.rpartition("@")[2])
        name = _norm_ws(name or "").strip("'\" ")
        if name and name.lower() != addr and "@" not in name:
            parsed = parse_display_name(name, loose=True)
            if parsed.is_person or addr not in self.email_person:
                self.add_parsed(parsed, addr)
        elif addr and addr not in self.email_person:
            guess = local_part_name(addr.partition("@")[0])
            if guess:
                self.email_person[addr] = self.add_person(*guess)

    def _collect_contacts(self, text: str) -> None:
        # "Jane Doe <jane@acme.com>", "Doe, Jane [mailto:jane@acme.com]"
        for m in re.finditer(r"(?P<n>[A-Z][\w'.\-]*(?:,?[ \t]+[A-Z][\w'.\-]*){0,3})[ \t]*[<\[(](?:mailto:)?"
                             r"(?P<a>[^\s<>\[\]()]+@[^\s<>\[\]()]+)[>\])]", text):
            self._collect_address(m.group("n"), m.group("a"))
        for m in EMAIL_RE.finditer(text):
            self._collect_address("", m.group(0))
        for m in URL_RE.finditer(text):
            host = re.sub(r"(?i)^(?:(?:https?|ftp)://)?", "", m.group(0)).split("/")[0].split("?")[0]
            self.note_domain(host.split("@")[-1].split(":")[0])
        for m in BARE_DOMAIN_RE.finditer(text):
            self.note_domain(m.group(1))

    def _collect_quoted_headers(self, body: str) -> None:
        for m in QHDR_RE.finditer(body):
            for piece in split_recipient_list(m.group("v")):
                name, addr = split_recipient(piece)
                self._collect_address(name, addr)

    def _collect_wrote_lines(self, body: str) -> None:
        for m in WROTE_RE.finditer(body):
            text = _norm_ws(re.sub(r"\n[ \t>]*", " ", m.group(0)))
            # The name follows the last date or time: "On Sep 21, 2026, at 4:02 PM, Jane Doe <jd@x.com> wrote:"
            mm = re.search(r"^.*(?:\b[AaPp]\.?[Mm]\.?\b|\d{4}|\d{1,2}:\d{2})[ ,]+"
                           r"(?P<n>[^<>]*?)[ ]*(?:<(?:mailto:)?(?P<a>[^>]+)>)?[ ]*wrote:", text)
            if mm:
                self._collect_address(mm.group("n") or "", mm.group("a") or "")

    def _collect_greetings(self, body: str) -> None:
        for m in GREETING_RE.finditer(body):
            names = re.sub(r"\s+(?:and|&)\s+", ",", m.group("names").strip())
            for piece in names.split(","):
                self._collect_short_name(piece)
        # A body that opens with a bare "Bob," line.
        first_line = next((line for line in body.splitlines() if line.strip()), "")
        mm = re.fullmatch(r"[ \t]*([A-Z][a-z]+)[ \t]*,[ \t]*", first_line)
        if mm and not SIGNOFF_RE.match(first_line.strip()):
            self._collect_short_name(mm.group(1))

    def _collect_short_name(self, piece: str) -> None:
        toks = piece.strip().split()
        if not toks or len(toks) > 3:
            return
        honor = toks[0].lower() in HONORIFICS
        toks = [t for t in toks if t.lower() not in HONORIFICS]
        if not toks or not all(looks_like_name_token(t) for t in toks):
            return
        if any(t.lower() in GREETING_SKIP or t.lower() in NOT_A_NAME for t in toks):
            return
        if honor and len(toks) == 1:
            self.add_person("", toks[0])
        elif len(toks) == 1:
            self.add_person(toks[0], "")
        else:
            self.add_parsed(parse_display_name(" ".join(toks), loose=True))

    def _collect_signatures(self, body: str) -> None:
        lines = body.splitlines()
        for i, line in enumerate(lines):
            text = strip_quote(line).strip()
            if len(text) > 70:
                continue
            m = SIGNOFF_RE.match(text)
            if not m:
                continue
            rest = m.group("rest").strip().strip(",.!")
            block: List[str] = []
            if rest:
                if parse_display_name(rest, loose=True).is_person:
                    block.append(rest)
                else:
                    continue
            j, blanks = i + 1, 0
            while j < len(lines) and len(block) < 14:
                t = strip_quote(lines[j]).strip()
                j += 1
                if not t:
                    blanks += 1
                    if blanks >= 2 and block:
                        break
                    continue
                if REPLY_BOUNDARY_RE.match(t):
                    break
                block.append(t)
                blanks = 0
            self._read_signature_block(block)

    def _link_nickname(self, token: str) -> None:
        """A sign-off like "Wendy" from Wendeline Achterkirk gets the sender's fake first name,
        so the scrubbed email still reads as one person."""
        sender = self._sender_first
        t, f = token.lower(), sender.lower()
        if not sender or t == f or len(t) < 2 or self.map.get("first", t):
            return
        if f.startswith(t) or (len(t) >= 3 and t[:3] == f[:3]):
            self.map.set("first", t, self.fake_token(sender, "first"))
            self._dirty()

    def _read_signature_block(self, block: List[str]) -> None:
        named = False
        for idx, line in enumerate(block):
            parts = [p.strip() for p in re.split(r"\s*[|\u2022\u00b7]\s*", line) if p.strip()]
            # A signature table turned into text: "Jane | Doe"
            if (not named and idx <= 2 and len(parts) >= 2 and all(len(p.split()) == 1 for p in parts[:2])
                    and all(looks_like_name_token(p) and p.lower() not in NOT_A_NAME for p in parts[:2])):
                self.add_person(parts[0], parts[1])
                named = True
                parts = parts[2:]
            for part in parts:
                if "@" in part or re.search(r"\d{3}", part) or URL_RE.search(part):
                    continue
                if not named and idx <= 2:
                    toks = part.split()
                    # A name split across two lines: "Jane" then "Doe".
                    if (len(toks) == 1 and idx + 1 < len(block) and len(block[idx + 1].split()) == 1
                            and looks_like_name_token(toks[0]) and looks_like_name_token(block[idx + 1])
                            and toks[0].lower() not in NOT_A_NAME and block[idx + 1].lower() not in NOT_A_NAME):
                        self.add_person(toks[0], block[idx + 1].strip())
                        named = True
                        continue
                    parsed = parse_display_name(part, loose=True)
                    if parsed.is_person and not parsed.company:
                        if parsed.first and not parsed.last:
                            self._link_nickname(parsed.first)
                        self.add_parsed(parsed)
                        named = True
                        continue
                if self._company_like(part):
                    self.add_company(part)

    def _company_like(self, text: str) -> bool:
        words = re.findall(r"[A-Za-z][A-Za-z'&.\-]*|&", text)
        if not words or len(words) > 7 or re.search(r"\d", text):
            return False
        if COMPANY_SUFFIX_RE.search(text):
            return True
        lowered = [w.lower().strip(".") for w in words]
        if any(w in TITLE_WORDS and w not in INDUSTRY_WORDS for w in lowered):
            return False
        if squash(text) and any(squash(text) == s or squash(text).startswith(s) and len(s) >= 4
                                for s in self.domain_stems):
            return True
        caps = [w for w in words if w[0].isupper() or w == "&"]
        if len(caps) != len(words):
            return False
        return any(w in INDUSTRY_WORDS for w in lowered) and any(is_distinctive(w) for w in words)

    def _collect_honorifics(self, text: str) -> None:
        for m in HONORIFIC_RE.finditer(text):
            a, b = m.group("a"), m.group("b")
            if b and a.lower() in COMMON_FIRST_NAMES and b.lower() not in NOT_A_NAME:
                self.add_person(a, b)
            elif a.lower() not in NOT_A_NAME:
                self.add_person("", a)

    def _collect_suffix_companies(self, text: str) -> None:
        # "Company: Acme Precision", "Ship to: Acme Precision"
        for m in LABELED_COMPANY_RE.finditer(text):
            value = m.group("v").strip()
            if 1 <= len(value.split()) <= 6 and all(w[0].isupper() or w in ("&", "and", "of") for w in value.split()):
                self.add_company(value)
        for m in COMPANY_SUFFIX_RE.finditer(text):
            before = text[: m.start()]
            line_start = before.rfind("\n") + 1
            seg = before[line_start:].rstrip(" ,\t")
            # "Sr. Buyer, Acme Precision Inc.": the name starts after the last comma or bar.
            seg = re.split(r"[,;:|()\[\]<>\u2022\u00b7]|\s-\s", seg)[-1]
            words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.\-]*|&", seg)
            run: List[str] = []
            for w in reversed(words):
                if w[0].isupper() or w[0].isdigit() and run or w in ("&", "and", "of", "de"):
                    run.insert(0, w)
                    if len(run) >= 5:
                        break
                else:
                    break
            while run and (run[0].lower() in ("and", "of", "&", "de") or
                           (run[0].lower().strip(".") in NOT_A_NAME and run[0].lower().strip(".") not in INDUSTRY_WORDS)):
                run.pop(0)
            if run and seg.endswith(run[-1]):
                self.add_company(" ".join(run) + " " + m.group("suf"))

    def _collect_domain_matches(self, text: str) -> None:
        """Capitalized words that spell a sender's domain ("Acme Precision" for
        acmeprecision.com) are that company's name."""
        if not self.domain_stems:
            return
        stems = {squash(s) for s in self.domain_stems if len(squash(s)) >= 4}
        for m in re.finditer(r"(?:[A-Z][A-Za-z0-9'\-]*|&)(?:[ \t]+(?:[A-Z][A-Za-z0-9'\-]*|&)){0,4}", text):
            words = m.group(0).split()
            for i in range(len(words)):
                for j in range(len(words), i, -1):
                    if squash("".join(words[i:j])) in stems:
                        self.add_company(" ".join(words[i:j]))
                        break

    def _collect_prose_names(self, body: str) -> None:
        for m in PROSE_NAME_RE.finditer(body):
            first, last = m.group("first"), m.group("last")
            fl, ll = first.lower(), last.lower()
            if fl not in COMMON_FIRST_NAMES or fl in NOT_A_NAME:      # "Virginia Beach" is a city
                continue
            if ll in NOT_A_NAME or ll in COMMON_FIRST_NAMES and ll in NAME_WORDS:
                continue
            if fl in NAME_WORDS and ll in NAME_WORDS:
                continue
            self.add_person(first, last, [m.group("mi")] if m.group("mi") else [])

    def _collect_cities(self, text: str) -> None:
        for rx in (CITY_ZIP_RE, CITY_CA_RE, CITY_ST_RE):
            for m in rx.finditer(text):
                words = m.group("city").split()
                while words and words[0].lower().strip(".") in NOT_A_NAME:
                    words.pop(0)
                if words:
                    self.fake_city(" ".join(words), m.group("state"))

    def _collect_context_names(self, text: str) -> None:
        """Two capitalized words where only a person fits: "cc Anatole Fairleigh", "ask
        Anatole Fairleigh", "Anatole Fairleigh will send"."""
        for rx in (CONTEXT_BEFORE_RE, CONTEXT_AFTER_RE):
            for m in rx.finditer(text):
                first, last = m.group("first"), m.group("last")
                if first.lower() in AMBIGUOUS - NAME_WORDS or last.lower() in NOT_A_NAME:
                    continue
                if first.lower() in NAME_WORDS and last.lower() in NAME_WORDS:
                    continue
                self.add_person(first, last, [m.group("mi")] if m.group("mi") else [])

    def _collect_parts(self, text: str) -> None:
        self.parts.update(self.find_parts(text))

    def find_parts(self, text: str) -> Counter:
        found: Counter = Counter()
        labeled_spans = []
        for m in PART_LABEL_RE.finditer(text):
            pn = m.group("pn").rstrip(".-/_")
            pn = re.split(r"_(?=Rev|rev|REV)", pn)[0]
            if re.search(r"\d", pn) and len(pn) >= 3 and not re.fullmatch(r"\d{1,2}", pn):
                found[pn] += 1
                self.part_labeled.add(pn.upper())
                labeled_spans.append((m.start("pn"), m.start("pn") + len(pn)))
        for m in PART_SHAPE_RE.finditer(text):
            pn = m.group(0)
            if any(a <= m.start() < b for a, b in labeled_spans) or self._not_a_part(text, m.start(), pn):
                continue
            found[pn] += 1
        return found

    @staticmethod
    def _not_a_part(text: str, start: int, pn: str) -> bool:
        head = re.match(r"[A-Z]+", pn)
        if head and head.group(0).lower() in SPEC_PREFIXES:
            return True
        if re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?|\d{5}-\d{4}|\d{3}-\d{3}-\d{4}|\d{3}-\d{4}", pn):
            return True
        if re.fullmatch(r"C\d{3,5}|H\d{3,4}|T\d{3,5}", pn):   # copper alloys (C360), tempers (H1150, T6511)
            return True
        if REFNUM_BEFORE_RE.search(text[max(0, start - 20):start]):
            return True
        return False

    # -- replacing -----------------------------------------------------------------------------

    def _count(self, kind: str, real: str, fake: str) -> None:
        real, fake = _norm_ws(real), _norm_ws(fake)
        if real and real != fake:
            self.stats[(kind, real, fake)] += 1
            self.leak_terms.add((kind, real))

    def scrub_text(self, text: str) -> str:
        if not text:
            return text
        slots = _Slots()
        s = _PH_RE.sub("", text)
        s = self._sub_manual(s, slots)
        s = MAILTO_QUERY_RE.sub(r"\1", s)
        s = self._sub_urls(s, slots)
        s = self._sub_emails(s, slots)
        s = self._sub_bare_domains(s, slots)
        s = self._sub_phones(s, slots)
        s = self._sub_addresses(s, slots)
        s = self._sub_company_forms(s, slots)
        s = self._sub_people(s, slots)
        s = self._sub_company_runs(s, slots)
        s = self._sub_cities(s, slots)
        s = self._sub_name_tokens(s, slots)
        if self.part_numbers:
            s = self._sub_parts(s, slots)
        return slots.restore(s)

    def scrub_filename(self, name: str) -> str:
        """Attachment names: scrub each piece between underscores and dots."""
        stem, dot, ext = name.rpartition(".")
        if not dot or len(ext) > 5 or not ext.isalnum():
            stem, dot, ext = name, "", ""
        pieces = re.split(r"(_+)", stem)
        out = "".join(p if p.startswith("_") else self.scrub_text(p) for p in pieces)
        return sanitize_filename(out + dot + ext)

    def _flex(self, real: str) -> str:
        return r"\s+".join(_esc(w) for w in real.split())

    def _sub_manual(self, s: str, slots: _Slots) -> str:
        for real, fake in self.manual_pairs():
            if not real:
                continue
            rx = re.compile(r"(?<![\w@.])" + self._flex(real) + r"(?![\w])", re.I)

            def rep(m: re.Match, fake: str = fake) -> str:
                out = match_case(m.group(0), fake)
                self._count("--map", m.group(0), out)
                return slots.put(out)
            s = rx.sub(rep, s)
        return s

    def _sub_urls(self, s: str, slots: _Slots) -> str:
        def rep(m: re.Match) -> str:
            url = m.group(0)
            trail = ""
            while url and url[-1] in ".,;:!?'\"":
                trail = url[-1] + trail
                url = url[:-1]
            scheme = re.match(r"(?i)^(?:(?:https?|ftp)://)?", url).group(0)
            rest = url[len(scheme):]
            hostport = re.split(r"[/?#]", rest, 1)[0]
            host = hostport.split("@")[-1].split(":")[0]
            path = rest[len(hostport):]
            fhost = self.fake_domain(host)
            if is_public_service(host) and path.strip("/"):
                fpath = "/s/" + _hex("url", url, n=10)
            else:
                fpath = "/" if path else ""
            fake = scheme + fhost + fpath
            self._count("url", url, fake)
            return slots.put(fake) + trail
        return URL_RE.sub(rep, s)

    def _sub_emails(self, s: str, slots: _Slots) -> str:
        def rep(m: re.Match) -> str:
            real = m.group(0)
            fake = self.fake_email(real)
            fake = match_case(real, fake) if real.isupper() else fake
            self._count("email", real, fake)
            return slots.put(fake)
        return EMAIL_RE.sub(rep, s)

    def _sub_bare_domains(self, s: str, slots: _Slots) -> str:
        def rep(m: re.Match) -> str:
            real = m.group(1)
            if not re.search(r"[a-z]", real.split(".")[-2] if "." in real else real, re.I):
                return real
            fake = match_case(real, self.fake_domain(real))
            self._count("domain", real, fake)
            return slots.put(fake)
        return BARE_DOMAIN_RE.sub(rep, s)

    def _phone_rep(self, slots: _Slots, text: str, num_start: int, num: str, ext: str, labeled: bool) -> Optional[str]:
        # "+44 (0)1457 ...": the "(0)" is dialed only at home, so it is kept as it is.
        num = num.replace("(0)", "\x00")
        digits = re.sub(r"\D", "", num)
        if not 7 <= len(digits) <= 15:
            return None
        if not labeled:
            before = text[max(0, num_start - 24):num_start]
            if re.search(r"(?i)\b(?:p/?n|part|dwg|drawing|item|model|po|p\.o\.|rfq|quote|order|"
                         r"invoice|job|lot|serial|s/n|rev)\b[\s#:.\-]*(?:no\.?\s*)?$", before):
                return None
        stripped = num.lstrip()
        cc = ""
        if stripped.startswith(("+", "00")):
            # The country code stays: +1, a written-apart "+353 1 ...", else two digits (+44, +49).
            if stripped.startswith("00"):
                digits = digits[2:]
            lead = re.match(r"(?:\+|00)[ ]?(\d+)", stripped).group(1)
            cc = "1" if lead.startswith("1") else lead if len(lead) <= 3 else lead[:2]
        fake_digits = self.fake_phone_digits(digits, cc)
        it = iter(("00" if stripped.startswith("00") else "") + fake_digits)
        fake = "".join(next(it, "0") if c.isdigit() else c for c in num).replace("\x00", "(0)")
        num = num.replace("\x00", "(0)")
        fext = ""
        if ext:
            ed = re.search(r"\d+$", ext).group(0)
            fe = str(_h("ext", ed) % (10 ** len(ed))).zfill(len(ed))
            fext = ext[: len(ext) - len(ed)] + fe
        self._count("phone", num + (ext or ""), fake + fext)
        self.leak_terms.add(("phone digits", digits[-7:]))
        return slots.put(fake + fext)

    def _sub_phones(self, s: str, slots: _Slots) -> str:
        def lab(m: re.Match) -> str:
            got = self._phone_rep(slots, m.string, m.start("num"), m.group("num"), m.group("ext") or "", True)
            if got is None:
                return m.group(0)
            return m.group(0)[: m.start("num") - m.start()] + got
        s = PHONE_LABEL_RE.sub(lab, s)

        def plain(m: re.Match) -> str:
            got = self._phone_rep(slots, m.string, m.start("num"), m.group("num"), m.group("ext") or "", False)
            return m.group(0) if got is None else got
        s = PHONE_NANP_RE.sub(plain, s)
        return PHONE_INTL_RE.sub(plain, s)

    def _sub_addresses(self, s: str, slots: _Slots) -> str:
        def street(m: re.Match) -> str:
            name = m.group("name")
            words = name.split()
            if any(w.upper() in ("AM", "PM") for w in words):
                return m.group(0)
            # "5 Axis Way", "2 Hole Pl": ordinary words before a suffix that is also a word or a
            # short abbreviation are not a street. "100 Main Way" still is.
            ordinary = all(w.lower().strip(".") in NOT_A_NAME - STREET_WORDS for w in words)
            if ordinary and not re.search(r"\d", name) and m.group("suf").lower().rstrip(".") in LOOSE_SUFFIXES:
                return m.group(0)
            real = m.group(0)
            key = _norm_ws(f"{m.group('num')} {name} {m.group('suf')}").lower().rstrip(".")
            fake_name = self.map.pick("street", key, FAKE_STREETS)
            num = str(100 + _h("num", key) % 9800)
            fake = f"{num} {fake_name} {m.group('suf')}"
            if m.group("unit"):
                fake += self._fake_unit(m.group("unit"))
            fake = match_case(real, fake) if real.upper() == real and any(c.isalpha() for c in real) else fake
            self._count("street", real, fake)
            for w in words:
                if len(w) > 3 and w.lower() not in NOT_A_NAME:
                    self.leak_terms.add(("street", f"{m.group('num')} {name}"))
                    break
            return slots.put(fake)
        s = STREET_RE.sub(street, s)

        def route(m: re.Match) -> str:
            real = m.group(0)
            key = real.lower()
            fake = f"{100 + _h('route', key) % 9800} {self.map.pick('street', key, FAKE_STREETS)} Rd"
            self._count("street", real, fake)
            return slots.put(fake)
        s = ROUTE_RE.sub(route, s)

        def pobox(m: re.Match) -> str:
            fake = f"{m.group('label')} {100 + _h('box', m.group('num')) % 9800}"
            self._count("po box", m.group(0), fake)
            return slots.put(fake)
        s = POBOX_RE.sub(pobox, s)

        def unit(m: re.Match) -> str:
            fake = f"{m.group('label')} {self._fake_unit_num(m.group('num'))}"
            self._count("suite", m.group(0), fake)
            return slots.put(fake)
        s = UNIT_RE.sub(unit, s)

        def city(m: re.Match, with_zip: bool = True) -> str:
            city_words = m.group("city").split()
            # "Ship To Springfield" -> the city is "Springfield"
            keep: List[str] = []
            while city_words and city_words[0].lower().strip(".") in NOT_A_NAME:
                keep.append(city_words.pop(0))
            if not city_words:
                return m.group(0)
            real_city = " ".join(city_words)
            state = m.group("state")
            fcity, fstate = self.fake_city(real_city, state)
            if len(state) > 2 and state.upper() not in PROVINCES:
                fstate = STATES.get(fstate, fstate)
            if m.re is CITY_CA_RE:
                fstate = state
            body = m.group(0)[len(m.group("city")):]
            out = body.replace(state, fstate, 1)
            if with_zip and m.groupdict().get("zip"):
                z = m.group("zip")
                if m.re is CITY_CA_RE:
                    fz = "".join(("ABCEGHJKLMNPRSTVXY"[_h("pc", z, str(i)) % 18] if c.isalpha()
                                  else str(_h("pc", z, str(i)) % 10) if c.isdigit() else c)
                                 for i, c in enumerate(z))
                else:
                    fz = str(10000 + _h("zip", z[:5]) % 89999).zfill(5)
                    if len(z) > 5:
                        fz += "-" + str(_h("zip4", z) % 10000).zfill(4)
                out = out[::-1].replace(z[::-1], fz[::-1], 1)[::-1]
            fake_city = match_case(real_city, fcity)
            self._count("city", m.group(0)[len(" ".join(keep)):].strip(), (fake_city + out).strip())
            self.leak_terms.add(("city", real_city))
            return (" ".join(keep) + " " if keep else "") + slots.put(fake_city + out)
        s = CITY_ZIP_RE.sub(city, s)
        s = CITY_CA_RE.sub(city, s)
        s = CITY_ST_RE.sub(lambda m: city(m, with_zip=False), s)
        return s

    def fake_city(self, city: str, state: str) -> Tuple[str, str]:
        abbr = state.upper() if len(state) == 2 else next((k for k, v in STATES.items() if v.lower() == state.lower()), state.lower())
        key = f"{_norm_ws(city).lower()}|{abbr}"
        fc = self.map.get("city", key)
        if not fc:
            used = self.map.used("city")
            start = _h("city", key)
            pair = FAKE_CITIES[start % len(FAKE_CITIES)]
            for i in range(len(FAKE_CITIES)):
                cand = FAKE_CITIES[(start + i) % len(FAKE_CITIES)]
                if f"{cand[0]}|{cand[1]}".lower() not in used:
                    pair = cand
                    break
            fc = f"{pair[0]}|{pair[1]}"
            self.map.set("city", key, fc)
            self._cache.pop("cities", None)
        fcity, fstate = fc.split("|")
        return fcity, fstate

    def _fake_unit_num(self, num: str) -> str:
        return "".join(str(_h("unit", num, str(i)) % 9 + 1) if c.isdigit() else c for i, c in enumerate(num))

    def _fake_unit(self, unit: str) -> str:
        m = re.search(r"[A-Z0-9][A-Z0-9\-]*$", unit)
        if not m:
            return unit
        return unit[: m.start()] + self._fake_unit_num(m.group(0))

    def _company_forms(self) -> List[List[str]]:
        if "companies" not in self._cache:
            forms = sorted(self.map.data["company_forms"], key=lambda k: (-len(k.split()), -len(k)))
            self._cache["companies"] = [f.split() for f in forms
                                        if len(f.split()) >= 2 or self.map.get("company", f)]
        return self._cache["companies"]

    def _sub_company_forms(self, s: str, slots: _Slots) -> str:
        """Whole company names ("Acme Precision Machining"), however they are spaced or cased."""
        squashed = squash(s)
        for words in self._company_forms():
            if squash("".join(words)) not in squashed:
                continue
            rx = self._cache.get(("form",) + tuple(words))
            if rx is None:
                body = r"([\s\-]*)".join("(" + re.escape(w) + ")" for w in words)
                rx = self._cache[("form",) + tuple(words)] = re.compile(_B + body + _E, re.I)
            def rep(m: re.Match, n: int = len(words)) -> str:
                groups = m.groups()
                real_words = [groups[2 * i] for i in range(n)]
                seps = [groups[2 * i + 1] for i in range(n - 1)]
                fake_words = self.fake_company_words(real_words)
                if len(fake_words) == n:
                    out = "".join(match_case(real_words[i], fake_words[i]) + (seps[i] if i < n - 1 else "")
                                  for i in range(n))
                else:
                    joiner = "" if seps and not any(seps) else " "
                    out = joiner.join(match_case(m.group(0), w) for w in fake_words)
                    if joiner == "":
                        out = match_case(m.group(0), "".join(fake_words))
                if out == m.group(0):
                    return m.group(0)
                self._count("company", m.group(0), out)
                return slots.put(out)
            s = rx.sub(rep, s)
        return s

    def _company_runs(self) -> List[Tuple[str, str]]:
        if "runs" not in self._cache:
            self._cache["runs"] = [
                (key, fake) for key, fake in sorted(self.map.data["company"].items(), key=lambda kv: -len(kv[0]))
                if not key.startswith("whole:") and isinstance(fake, str) and all(is_distinctive(w) for w in key.split())]
        return self._cache["runs"]

    def _company_run_pattern(self, key: str) -> re.Pattern:
        rx = self._cache.get(("run", key))
        if rx is None:
            words = key.split()
            if len(words) == 1 and (key in self.map.data["company_weak"] or key in AMBIGUOUS or len(key) <= 3):
                # A common word ("Summit", "Star"), or a guess from a domain, is replaced only
                # when capitalized.
                rx = re.compile(_B + "(?:" + re.escape(key[:1].upper() + key[1:]) + "|" + re.escape(key.upper()) + ")" + _E)
            else:
                rx = re.compile(_B + r"[\s\-]*".join(re.escape(w) for w in words) + _E, re.I)
            self._cache[("run", key)] = rx
        return rx

    def _sub_company_runs(self, s: str, slots: _Slots) -> str:
        """The distinctive words of known companies, alone ("Acme will send the PO")."""
        squashed = squash(s)
        for key, fake in self._company_runs():
            if squash(key) not in squashed:
                continue
            rx = self._company_run_pattern(key)
            def rep(m: re.Match, fake: str = fake) -> str:
                real = m.group(0)
                out = match_case(real, fake)
                self._count("company", real, out)
                return slots.put(out)
            s = rx.sub(rep, s)
        return s

    def _city_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Known real cities alone in the text ("our Tempe plant"), capitalized only."""
        if "cities" not in self._cache:
            pats = []
            names: Dict[str, str] = {}
            for key, fake in self.map.data["city"].items():
                city = key.split("|")[0]
                if any(w in AMBIGUOUS for w in city.split()) or len(city) < 4:
                    continue
                names.setdefault(city, fake.split("|")[0])
            for city in sorted(names, key=len, reverse=True):
                title = " ".join(w[:1].upper() + w[1:] for w in city.split())
                body = "(?:" + r"\s+".join(re.escape(w) for w in title.split()) + "|" + \
                    r"\s+".join(re.escape(w.upper()) for w in title.split()) + ")"
                pats.append((re.compile(_B + body + _E), names[city]))
            self._cache["cities"] = pats
        return self._cache["cities"]

    def _sub_cities(self, s: str, slots: _Slots) -> str:
        for rx, fake in self._city_patterns():
            def rep(m: re.Match, fake: str = fake) -> str:
                out = match_case(m.group(0), fake)
                self._count("city", m.group(0), out)
                return slots.put(out)
            s = rx.sub(rep, s)
        return s

    def _people_list(self) -> List[Dict[str, Any]]:
        if "people" not in self._cache:
            self._cache["people"] = sorted(
                self.map.data["people"].values(),
                key=lambda p: -(len(p["first"]) + len(p["last"]) + sum(len(x) for x in p["middle"])))
        return self._cache["people"]

    def _person_pattern(self, p: Dict[str, Any], which: str) -> re.Pattern:
        """which: "both" (every form), "last" ("J. Doe" only), "first" ("Jane D." only). Compiled
        when first needed, since most people in a big map are not in a given email."""
        key = ("person", p["first"], p["last"], which)
        rx = self._cache.get(key)
        if rx is None:
            f, l = _esc(p["first"]), r"\s+".join(_esc(x) for x in p["last"].split())
            mids = [_esc(x.rstrip(".")) for x in p["middle"] if len(x.rstrip(".")) > 1]
            mid = r"(?:\s+(?:[A-Za-z]\.?" + "".join("|" + x for x in mids) + r"))*"
            initial_last = re.escape(p["first"][0]) + r"\.\s*" + l
            first_initial = f + r"\s+" + re.escape(p["last"][0]) + r"\."
            if which == "both":
                alts = [f + mid + r"\s+" + l, l + r",\s*" + f + r"(?:\s+[A-Za-z]\.)?", initial_last, first_initial]
            else:
                alts = [initial_last] if which == "last" else [first_initial]
            rx = re.compile(_B + "(?:" + "|".join(alts) + ")" + _E, re.I)
            self._cache[key] = rx
        return rx

    def _sub_people(self, s: str, slots: _Slots) -> str:
        """Full names (and "Last, First", "J. Doe", "Jane D.") of known people, whatever the
        line breaks between the parts."""
        present = set(re.findall(r"[^\W\d_]+", s.lower().replace("\u2019", "'")))
        for p in self._people_list():
            has_first = set(re.findall(r"[^\W\d_]+", p["first"].lower())) <= present
            has_last = set(re.findall(r"[^\W\d_]+", p["last"].lower())) <= present
            if not has_first and not has_last:
                continue
            rx = self._person_pattern(p, "both" if has_first and has_last else "last" if has_last else "first")
            first, last = p["first"], p["last"]
            ff, fl = self.fake_token(first, "first"), self.fake_token(last, "last")
            mids = {x.lower().rstrip("."): self.fake_token(x, "first") for x in p["middle"] if len(x.rstrip(".")) > 1}

            def rep(m: re.Match) -> str:
                span = m.group(0)
                lastwords = last.lower().split()
                fakelast = fl.split()[:1] if len(fl.split()) else [fl]

                core = [x for x in lastwords if x not in NAME_PARTICLES]

                def tok(t: re.Match) -> str:
                    w = t.group(0)
                    lw = w.lower()
                    if lw == first.lower():
                        return match_case(w, ff)
                    if lw in NAME_PARTICLES and lw in lastwords:
                        return w                     # "dos Santos" -> "dos Walmsley"
                    if lw in lastwords:
                        # One surname word gets the surname's fake; a two-word surname
                        # ("Garcia Lopez") gets one fake per word.
                        return match_case(w, " ".join(fakelast) if len(core) == 1 else self.fake_token(w, "last"))
                    if lw in mids:
                        return match_case(w, mids[lw])
                    if len(w) == 1:
                        if lw == first[0].lower() and span.lower().startswith(lw):
                            return match_case(w, ff[0])
                        if lw == last[0].lower() and not span.lower().startswith(lw):
                            return match_case(w, fl[0])
                        return match_case(w, "ABCDEFGHJKLMNPRSTW"[_h("mi", w, first) % 18])
                    return w
                out = re.sub(r"[^\W\d_]+(?:['\u2019\-][^\W\d_]+)*", tok, span.replace("\u2019", "'"))
                out = re.sub(r"\s{2,}", " ", out) if not re.search(r"\n", span) else out
                self._count("person", span, out)
                return slots.put(out)
            s = rx.sub(rep, s)
        return s

    def _token_patterns(self) -> Tuple[Optional[re.Pattern], Optional[re.Pattern]]:
        if "tokens" not in self._cache:
            plain, ambiguous = [], []
            keys = set(self.map.data["first"]) | set(self.map.data["last"])
            for k in sorted(keys, key=len, reverse=True):
                if len(k) < 2:
                    continue
                (ambiguous if (k in AMBIGUOUS or len(k) <= 2) else plain).append(k)
            rx = amb = None
            if plain:
                rx = re.compile(_B + "(" + "|".join(self._flex(k) for k in plain) + ")" + _E, re.I)
            if ambiguous:
                forms = sorted({f for k in ambiguous for f in (k[:1].upper() + k[1:], k.upper())}, key=len, reverse=True)
                amb = re.compile(_B + "(" + "|".join(self._flex(f) for f in forms) + ")" + _E)
            self._cache["tokens"] = (rx, amb)
        return self._cache["tokens"]

    def _sub_name_tokens(self, s: str, slots: _Slots) -> str:
        """Name words left after the full names: first names alone, surnames alone,
        possessives."""
        rx, amb = self._token_patterns()
        if rx is not None:
            def rep(m: re.Match) -> str:
                real = m.group(0)
                fake = match_case(real, self._token_fake(real))
                self._count("name", real, fake)
                return slots.put(fake)
            s = rx.sub(rep, s)
        if amb is not None:
            def arep(m: re.Match) -> str:
                if not name_context(m.string, m.start(), m.end()):
                    return m.group(0)
                real = m.group(0)
                fake = match_case(real, self._token_fake(real))
                self._count("name", real, fake)
                return slots.put(fake)
            s = amb.sub(arep, s)
        return s

    def _token_fake(self, token: str) -> str:
        key = _norm_ws(token).lower()
        return self.map.get("first", key) or self.map.get("last", key) or self.fake_token(key)

    def _sub_parts(self, s: str, slots: _Slots) -> str:
        parts = sorted(set(self.parts) | set(self.map.data["part"]), key=len, reverse=True)
        if not parts:
            return s
        alts = []
        for p in parts:
            alts.append("".join(re.escape(c) if c.isalnum() else r"[-.\s]?" for c in p))
        rx = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(alts) + r")(?![A-Za-z0-9])", re.I)

        def rep(m: re.Match) -> str:
            real = m.group(0)
            canon = next((p for p in parts if squash(p) == squash(real)), real)
            fake = self.fake_part(canon)
            if canon != real and squash(canon) == squash(real):
                it = iter(c for c in fake if c.isalnum())
                fake = "".join(next(it, "") if c.isalnum() else c for c in real)
            self._count("part number", real, fake)
            return slots.put(fake)
        return rx.sub(rep, s)


def name_context(text: str, start: int, end: int) -> bool:
    """True where a word that is also an ordinary word is used as a name."""
    before = text[max(0, start - 40):start]
    after = text[end:end + 40]
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start: len(text) if line_end < 0 else line_end]
    if re.fullmatch(r"[ \t>]*[-~]*[ \t]*" + re.escape(text[start:end]) + r"[ \t,.!]*", line):
        return True                                      # alone on a line (a sign-off)
    if re.match(r"['\u2019]s\b", after):
        return True                                      # possessive
    if _PH_RE.match(after.lstrip(" \t")[:1] or "") or _PH_RE.search(before[-2:] or ""):
        return True                                      # next to a replaced name
    if re.search(r"(?i)\b(?:hi|hello|hey|dear|thanks|thank you|thx|cheers|ask|call|email|e-mail|cc|"
                 r"tell|contact|with|from|per|to|attn:?|attention:?|mr\.?|mrs\.?|ms\.?|dr\.?|and|or|"
                 r"regards|best|sincerely)[ \t,]+$", before):
        return True
    if re.match(r"[ \t]+(?:said|says|mentioned|asked|wants|called|emailed|sent|wrote|told|"
                r"suggested|requested|confirmed|approved|noted|will\s+(?:send|call|email|follow))\b", after):
        return True
    if re.match(r"[ \t]*,[ \t]*(?:\n|$|please|can|could|would|we|i|the|attached)", after, re.I) and \
            (line.strip().startswith(text[start:end])):
        return True                                      # "Mark, please see attached"
    return False


def company_core(name: str) -> List[str]:
    """'The Acme Precision Machining, Inc.' -> ['Acme', 'Precision', 'Machining']"""
    text = _norm_ws(name.replace(",", " "))
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.\-]*|&", text)
    while words and words[-1].lower() in _SUFFIX_WORDS | {"pty", "s.a", "de", "c.v."}:
        words.pop()
    words = [w.rstrip(".") if not re.fullmatch(r"(?:[A-Za-z]\.)+", w) else w for w in words]
    while words and words[0].lower() in ("the", "and", "&", "of"):
        words.pop(0)
    while words and words[-1].lower() in ("and", "&", "of", "the"):
        words.pop()
    return [w for w in words if w]


def is_distinctive(word: str) -> bool:
    """A company word that identifies it: not a trade word, a job title, or an ordinary word."""
    lw = word.lower().strip(".")
    return (len(squash(lw)) >= 2 and lw not in INDUSTRY_WORDS and lw not in TITLE_WORDS
            and lw not in COMMON_WORDS and lw not in ("&", "and", "of", "the") and not lw.isdigit())


def distinctive_runs(words: Sequence[str]) -> List[List[str]]:
    runs: List[List[str]] = []
    cur: List[str] = []
    for w in words:
        if is_distinctive(w):
            cur.append(w)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


_STEM_TAILS = sorted({w for w in INDUSTRY_WORDS if w.isalpha() and len(w) >= 2} |
                     {"mfg", "inc", "corp", "co", "llc", "usa", "us", "intl", "grp", "sys", "eng", "ind", "med",
                      "tech", "aero", "labs", "mach", "prec", "precise", "cnc", "hq"}, key=len, reverse=True)


def split_stem(stem: str) -> List[str]:
    """acmeprecision -> ['acme', 'precision']; brightbore -> ['brightbore']."""
    s = stem.lower()
    tail: List[str] = []
    changed = True
    while changed:
        changed = False
        for t in _STEM_TAILS:
            if s.endswith(t) and len(s) - len(t) >= 3:
                tail.insert(0, t)
                s = s[: -len(t)]
                changed = True
                break
    return [s] + tail


def fake_first_for(key: str) -> str:
    return FAKE_FIRST[_h("ff", key) % len(FAKE_FIRST)]


def fake_last_for(key: str) -> str:
    return FAKE_LAST[_h("fl", key) % len(FAKE_LAST)]


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\x00-\x1f\x7f;/\\]", " ", name)
    name = _norm_ws(name)[:120]
    return name or "attachment"


# ---------------------------------------------------------------------------------------------
# What still looks identifying

CHECK_TITLE_PAIR_RE = re.compile(r"(?<![\w])(?=([A-Z][a-z]{1,20})[ \t]+([A-Z][a-z]{2,20})(?![\w]))")
CHECK_PHONEISH_RE = re.compile(r"(?<![\w.])\+?\(?\d[\d \t().\-]{6,20}\d(?![\w])")


def check_these(scrubber: Scrubber, fields: Sequence[Tuple[str, str]]) -> List[str]:
    """Anything in the scrubbed output that still looks identifying, with where it is."""
    fakes = scrubber.map.fakes()
    fake_emails = set(scrubber.map.data["email"].values())
    fake_phones = {str(v) for v in scrubber.map.data["phone"].values()}
    fake_domains = {v for k, v in scrubber.map.data["domain"].items() if not k.endswith("|stem")}
    out: List[str] = []
    seen: Set[str] = set()

    def add(where: str, msg: str) -> None:
        item = f"{where}: {msg}"
        if item not in seen:
            seen.add(item)
            out.append(item)

    # 1. Real values this run replaced somewhere, or the map knows, still present. One pattern for
    # all of them, so a big map stays fast.
    weak = scrubber.map.data["company_weak"]
    anycase: Dict[str, str] = {}     # lowercase term -> kind
    capital: Dict[str, str] = {}     # "Mark" / "MARK" -> kind, for words that are also ordinary words
    digit_terms: Set[str] = set()
    checked = {"name", "person", "company", "email", "domain", "--map", "street", "city"}
    if scrubber.part_numbers:
        checked.add("part number")

    def term(kind: str, value: str) -> None:
        low = _norm_ws(value).lower()
        if len(low) < 3 or (low in fakes and kind != "email"):
            return
        if kind in ("name", "company") and (low in AMBIGUOUS or len(low) <= 3 or low in weak):
            capital.setdefault(low[:1].upper() + low[1:], kind)
            capital.setdefault(low.upper(), kind)
        else:
            anycase.setdefault(low, kind)

    for kind, value in scrubber.leak_terms:
        if kind == "phone digits":
            digit_terms.add(value)
        elif kind in checked:
            term(kind, value)
    for section in ("first", "last"):
        for k in scrubber.map.data[section]:
            term("name", k)
    for k in scrubber.map.data["company"]:
        # A name of generic words only ("Valve Works") can be left around a fake stem.
        if not k.startswith("whole:") and distinctive_runs(k.split()):
            term("company", k)
    for k in scrubber.map.data["email"]:
        term("email", k)
    for stem in scrubber.domain_stems:
        if len(stem) >= 4:
            term("domain", stem)
    for k in scrubber.map.data["manual"]:
        term("--map", k)

    def alternation(words: Iterable[str]) -> str:
        return "|".join(r"\s+".join(_esc(w) for w in t.split()) for t in sorted(words, key=len, reverse=True))

    any_rx = re.compile(_B + "(?:" + alternation(anycase) + ")" + _E, re.I) if anycase else None
    cap_rx = re.compile(_B + "(?:" + alternation(capital) + ")" + _E) if capital else None
    for where, text in fields:
        if not text:
            continue
        lines = text.splitlines() or [text]
        for n, line in enumerate(lines, 1):
            loc = f"{where} line {n}" if len(lines) > 1 else where
            if any_rx is not None:
                for m in any_rx.finditer(line):
                    kind = anycase.get(_norm_ws(m.group(0)).lower(), "value")
                    add(loc, f"real {kind} \"{m.group(0)}\" still here")
            if cap_rx is not None:
                for m in cap_rx.finditer(line):
                    kind = capital.get(m.group(0), "name")
                    add(loc, f"{kind} word \"{m.group(0)}\" still here (it is also an ordinary word; check the use)")
            dline = re.sub(r"\D", "", line)
            for d in digit_terms:
                if d and d in dline:
                    add(loc, f"digits of a real phone number ({d[:3]}-{d[3:]}) still here")
    # 2. Patterns that look identifying and are not fakes.
    for where, text in fields:
        if not text:
            continue
        lines = text.splitlines() or [text]
        for n, line in enumerate(lines, 1):
            loc = f"{where} line {n}" if len(lines) > 1 else where
            for m in EMAIL_RE.finditer(line):
                a = m.group(0).lower()
                if a not in fake_emails and not is_reserved_domain(a.rpartition("@")[2]):
                    add(loc, f"email address \"{m.group(0)}\"")
            for m in URL_RE.finditer(line):
                host = re.sub(r"(?i)^(?:(?:https?|ftp)://)?", "", m.group(0)).split("/")[0].split(":")[0].lower()
                _, reg = registrable(host)
                if ".".join(reg) not in fake_domains and not is_public_service(host) and not is_reserved_domain(host):
                    add(loc, f"web address \"{m.group(0)}\"")
            for m in CHECK_PHONEISH_RE.finditer(line):
                d = re.sub(r"\D", "", m.group(0).replace("(0)", ""))
                if 10 <= len(d) <= 15 and "55501" not in d and not any(d.endswith(f) or f.endswith(d) for f in fake_phones) \
                        and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", m.group(0).strip()):
                    add(loc, f"number that looks like a phone \"{m.group(0).strip()}\"")
            for m in CHECK_TITLE_PAIR_RE.finditer(line):
                a, b = m.group(1), m.group(2)
                la, lb = a.lower(), b.lower()
                if la in fakes or lb in fakes:
                    continue
                if la in COMMON_FIRST_NAMES and la not in AMBIGUOUS and lb not in NOT_A_NAME:
                    add(loc, f"possible name \"{a} {b}\"")
                elif not ({la, lb} & (AMBIGUOUS | COMMON_FIRST_NAMES)):
                    add(loc, f"possible name or company \"{a} {b}\"")
            for m in re.finditer(r"(?<![\w])[A-Z][a-z]{2,}(?![\w])", line):
                w = m.group(0).lower()
                if w in COMMON_FIRST_NAMES and w not in AMBIGUOUS and w not in fakes:
                    add(loc, f"possible first name \"{m.group(0)}\"")
            for m in HONORIFIC_RE.finditer(line):
                if m.group("a").lower() not in fakes:
                    add(loc, f"possible name \"{m.group(0)}\"")
            for m in COMPANY_SUFFIX_RE.finditer(line):
                before = re.findall(r"[A-Za-z][A-Za-z'&\-]*", line[: m.start()])
                if before and before[-1][0].isupper() and before[-1].lower() not in fakes and is_distinctive(before[-1]) \
                        and before[-1].lower() not in NOT_A_NAME:
                    add(loc, f"possible company \"{before[-1]} {m.group(0)}\"")
            for m in STREET_RE.finditer(line):
                words = [w.lower() for w in m.group("name").split()]
                if not any(w in fakes for w in words):
                    add(loc, f"possible street address \"{m.group(0).strip()}\"")
            for rx in (CITY_ZIP_RE, CITY_CA_RE):
                for m in rx.finditer(line):
                    if not any(w.lower() in fakes for w in m.group("city").split()):
                        add(loc, f"possible city and postal code \"{m.group(0).strip()}\"")
    return out


# ---------------------------------------------------------------------------------------------
# Writing the scrubbed email

def build_eml(*, from_name: str, from_email: str, to: Sequence[Tuple[str, str]],
              cc: Sequence[Tuple[str, str]], subject: str, date: Optional[str], message_id: str,
              body: str, dropped: Sequence[str], kept: Sequence[Dict[str, Any]], boundary: str) -> bytes:
    msg = EmailMessage(policy=POLICY)
    msg["From"] = Address(display_name=from_name, addr_spec=from_email) if from_email else from_name
    if to:
        msg["To"] = [Address(display_name=n, addr_spec=a) for n, a in to]
    if cc:
        msg["Cc"] = [Address(display_name=n, addr_spec=a) for n, a in cc]
    msg["Subject"] = subject
    when = parse_iso(date)
    if when is not None:
        msg["Date"] = email_utils.format_datetime(when)
    msg["Message-ID"] = f"<{message_id}>"
    msg["X-Scrubbed-By"] = "tools/scrub_email.py"
    if dropped:
        msg["X-Scrubbed-Attachments"] = "; ".join(dropped)
    text = body if body.endswith("\n") else body + "\n"
    # 8bit keeps the file readable; quoted-printable only when a line is too long for 8bit.
    cte = "8bit" if all(len(line.encode("utf-8")) <= 900 for line in text.split("\n")) and "\r" not in text \
        else "quoted-printable"
    msg.set_content(text, subtype="plain", charset="utf-8", cte=cte)
    for att in kept:
        ctype = att.get("content_type") or "application/octet-stream"
        maintype, _, subtype = ctype.partition("/")
        msg.add_attachment(att["data"], maintype=maintype or "application",
                           subtype=subtype or "octet-stream", filename=att["name"])
    if kept:
        msg.set_boundary(boundary)
    return msg.as_bytes(policy=POLICY)


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt


def slugify(text: str, limit: int = 48) -> str:
    s = re.sub(r"^(?:(?:re|fw|fwd|aw|wg)\s*:\s*)+", "", fold(text).lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s[:limit].rstrip("-")) or "email"


def email_id(em: Dict[str, Any]) -> str:
    """Stable id of a real email, from its Message-ID, else its sender, date, subject, and body."""
    if em.get("message_id"):
        return _hex("mid", em["message_id"].strip().lower(), n=8)
    return _hex("em", (em.get("from_email") or "").lower(), em.get("date") or "", em.get("subject") or "",
                (em.get("body") or "")[:2000], n=8)


# ---------------------------------------------------------------------------------------------
# Manifest

def load_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SystemExit(f"{path} is not valid JSON ({exc}); fix it first.")
        if isinstance(data, dict):
            data.setdefault("files", [])
            return data
    return {"about": NEW_MANIFEST_ABOUT, "files": []}


def update_manifest(data: Dict[str, Any], entry: Dict[str, Any], old_names: Iterable[str] = ()) -> None:
    files = data.setdefault("files", [])
    drop = set(old_names) - {entry["file"]}
    files[:] = [f for f in files if not (isinstance(f, dict) and f.get("file") in drop
                                         and f.get("source") == "scrubbed")]
    for i, f in enumerate(files):
        if isinstance(f, dict) and f.get("file") == entry["file"]:
            files[i] = entry
            return
    files.append(entry)


def manifest_text(data: Dict[str, Any], indent: int = 1) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False) + "\n"


def manifest_indent(path: Path) -> int:
    """The indent the manifest already uses (tools/make_email_fixtures.py writes 1), so a scrub
    run does not reformat the generated entries."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 1
    m = re.search(r'\n( +)"', text)
    return len(m.group(1)) if m else 1


# ---------------------------------------------------------------------------------------------
# Running

def expand_inputs(inputs: Sequence[str]) -> Tuple[List[Path], List[str]]:
    """Files, folders (every .eml, .msg, .zip inside), and wildcards (the Windows shell does not
    expand them)."""
    found: List[Path] = []
    problems: List[str] = []
    for raw in inputs:
        paths = [Path(p) for p in sorted(glob.glob(raw))] if any(c in raw for c in "*?[") else [Path(raw)]
        if not paths:
            problems.append(f"{raw}: nothing matches")
        for p in paths:
            if p.is_dir():
                kids = sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in EMAIL_EXTS)
                if not kids:
                    problems.append(f"{p}: no .eml, .msg, or .zip files inside")
                found.extend(kids)
            elif p.is_file():
                found.append(p)
            else:
                problems.append(f"{p}: not found")
    unique: List[Path] = []
    for p in found:
        if p not in unique:
            unique.append(p)
    return unique, problems


def _load_mailfile():
    try:
        import mailfile  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"mailfile.py is needed to read emails ({exc}).")
    return mailfile


def parse_map_args(pairs: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for p in pairs or ():
        real, sep, fake = p.partition("=")
        if not sep or not real.strip() or not fake.strip():
            raise SystemExit(f'--map needs "Real=Fake", got {p!r}')
        out.append((real.strip(), fake.strip()))
    return out


def run(argv: Optional[Sequence[str]] = None, stdout=None) -> int:
    out = stdout or sys.stdout
    ap = argparse.ArgumentParser(
        prog="scrub_email.py",
        description="Turn real emails (.eml, .msg, .zip) into fictional test emails for tests/emails/.")
    ap.add_argument("inputs", nargs="+", metavar="INPUT", help=".eml, .msg, or .zip files, or folders of them")
    ap.add_argument("--lane", required=True, choices=LANES, help="the lane the email should route to")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--rfq", dest="is_rfq", action="store_const", const=True, help="the email is an RFQ")
    group.add_argument("--not-rfq", dest="is_rfq", action="store_const", const=False, help="the email is not an RFQ")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="folder for the .eml files and manifest.json "
                    "(default tests/emails)")
    ap.add_argument("--map", action="append", default=[], metavar='"Real=Fake"',
                    help="also replace this text (remembered for later runs); repeat as needed")
    ap.add_argument("--keep-attachments", action="store_true",
                    help="keep attachments as they are (NOT scrubbed; check each by hand)")
    ap.add_argument("--part-numbers", action="store_true", help="replace part numbers too")
    ap.add_argument("--dry-run", action="store_true", help="print the report and write nothing")
    ap.add_argument("--map-file", default=str(DEFAULT_MAP),
                    help="the real-to-fake map (default private_emails/.scrub_map.json; it holds real "
                         "names, so keep it out of git)")
    args = ap.parse_args(argv)

    is_rfq = args.is_rfq
    if is_rfq is None:
        if args.lane not in LANE_IS_RFQ:
            ap.error(f"--lane {args.lane} holds RFQs and other email: add --rfq or --not-rfq")
        is_rfq = LANE_IS_RFQ[args.lane]
    manual = parse_map_args(args.map)
    mailfile = _load_mailfile()
    files, problems = expand_inputs(args.inputs)
    out_dir = Path(args.out)
    map_path = Path(args.map_file)
    smap = ScrubMap(map_path)
    scrubber = Scrubber(smap, manual, part_numbers=args.part_numbers)

    # Read everything first, so a name seen in one email is known when scrubbing all of them.
    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    skipped: List[str] = list(problems)
    for path in files:
        try:
            data = path.read_bytes()
        except OSError as exc:
            skipped.append(f"{path}: {exc}")
            continue
        res = mailfile.load(path.name, data)
        for sk in res.get("skipped") or []:
            skipped.append(f"{path.parent / sk.get('source', path.name)}: {sk.get('reason', 'skipped')}")
        for em in res.get("emails") or []:
            loaded.append((path, em))
    for _, em in loaded:
        scrubber.collect(em)
    scrubber.finish_collect()

    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    indent = manifest_indent(manifest_path)
    before_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
    reports: List[str] = []
    written = unchanged = 0
    kept_total = 0
    for path, em in loaded:
        src = em.get("source") or path.name
        if em.get("container_only"):
            reports.append(f"== {src}\n   A forward bundle: the emails attached to it are written instead.\n")
            continue
        scrubber.stats = Counter()
        scrubber.leak_terms = set()
        res = scrub_one(scrubber, em, keep_attachments=args.keep_attachments)
        eid = email_id(em)
        fname = f"scrubbed_{slugify(res['subject'])}_{eid}.eml"
        target = out_dir / fname
        old_names = [f.get("file") for f in manifest.get("files", [])
                     if isinstance(f, dict) and f.get("source") == "scrubbed"
                     and str(f.get("file", "")).startswith("scrubbed_") and str(f.get("file", "")).endswith(f"_{eid}.eml")]
        previous = next((f for f in manifest.get("files", []) if isinstance(f, dict) and f.get("file") == fname), None)
        # Read it back the way the importer will, so the manifest lists what load() sees.
        back = mailfile.load(fname, res["eml"])
        back_em = (back.get("emails") or [{}])[0]
        roundtrip = []
        if (back_em.get("subject") or "") != res["subject"].strip():
            roundtrip.append("subject")
        if (back_em.get("from_email") or "") != res["from_email"].lower():
            roundtrip.append("sender")
        if _norm_body(back_em.get("body") or "") != _norm_body(res["body"]):
            roundtrip.append("body")
        # The contract has load() list each X-Scrubbed-Attachments name as a file-name-only
        # attachment; the manifest says what the file holds either way.
        attachments = res["dropped"] + [k["name"] for k in res["kept"]]
        back_names = [a.get("name") for a in back_em.get("attachments") or []]
        if sorted(back_names) == sorted(attachments):
            attachments = back_names          # load()'s order, which tests/test_import.py compares
        else:
            roundtrip.append("attachment list")
        entry = {"file": fname, "source": "scrubbed", "reviewed": False,
                 "emails": [{"subject": back_em.get("subject", res["subject"]),
                             "from_email": back_em.get("from_email", res["from_email"]),
                             "attachments": attachments, "lane": args.lane, "is_rfq": is_rfq}]}
        same = target.exists() and target.read_bytes() == res["eml"]
        # A file that did not change keeps its review, unless its lane or answers changed.
        if same and previous and previous.get("reviewed") is True and previous.get("emails") == entry["emails"]:
            entry["reviewed"] = True
        status = ("would write (no change)" if same else "would write") if args.dry_run else "wrote"
        if not args.dry_run:
            if same:
                status = "unchanged" + (", still reviewed" if entry["reviewed"] else "")
                unchanged += 1
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                target.write_bytes(res["eml"])
                written += 1
            for old in old_names:
                if old != fname and (out_dir / old).exists():
                    (out_dir / old).unlink()
        update_manifest(manifest, entry, old_names)
        kept_total += len(res["kept"])
        reports.append(format_report(src, target, status, res, scrubber, args, is_rfq, roundtrip))

    if not args.dry_run and loaded:
        text = manifest_text(manifest, indent)
        if text != before_manifest:
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(text, encoding="utf-8")
        smap.save()

    print("Scrub report" + ("  (DRY RUN: nothing was written)" if args.dry_run else ""), file=out)
    print(f"Map: {map_path}", file=out)
    print("", file=out)
    for r in reports:
        print(r, file=out)
    if skipped:
        print("Skipped:", file=out)
        for s in skipped:
            print(f"   {s}", file=out)
        print("", file=out)
    n = sum(1 for _, em in loaded if not em.get("container_only"))
    if args.dry_run:
        print(f"{n} email(s) scrubbed in memory. Nothing written.", file=out)
    else:
        print(f"{n} email(s): {written} written, {unchanged} unchanged, into {out_dir}. "
              f"Manifest: {manifest_path}", file=out)
        print("Open every file, fix anything on its \"check these\" list (rerun with --map), then set "
              "\"reviewed\": true in the manifest.", file=out)
    if kept_total:
        # On stderr, so it shows even when the report goes to a file.
        print(f"\n!!! WARNING: --keep-attachments copied {kept_total} attachment(s) UNSCRUBBED. Drawings, "
              "forms, and PDFs carry names, logos, addresses, and title blocks. Open every one and "
              "check it by hand before you commit. !!!\n", file=sys.stderr)
    return 0 if loaded or not skipped else 1


def _norm_body(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").strip().split("\n"))


def scrub_one(scrubber: Scrubber, em: Dict[str, Any], keep_attachments: bool = False) -> Dict[str, Any]:
    """Scrub one parsed email. Returns the .eml bytes and what went into it."""
    dashes = 0
    em = dict(em)
    for field in ("subject", "body", "from_name"):
        em[field], n = plain_dashes(em.get(field) or "")
        dashes += n
    em["to"] = [plain_dashes(v)[0] for v in em.get("to") or []]
    em["cc"] = [plain_dashes(v)[0] for v in em.get("cc") or []]
    em["attachments"] = [dict(a, name=plain_dashes(a.get("name") or "")[0]) for a in em.get("attachments") or []]
    from_email = (em.get("from_email") or "").strip().lower()
    from_name = em.get("from_name") or ""
    if from_email:
        f_email = scrubber.fake_email(from_email)
    else:
        # Exchange senders with no SMTP address still get one, on a fake domain.
        f_email = scrubber.fake_email((squash(from_name) or "sender") + "@unknown-sender.invalid.scrub")
    f_name = scrubber.scrub_text(from_name) if from_name else ""

    def people(values: Sequence[str]) -> List[Tuple[str, str]]:
        out = []
        for v in values or []:
            name, addr = split_recipient(v)
            fa = scrubber.fake_email(addr) if addr else ""
            fn = scrubber.scrub_text(name) if name else ""
            if fa:
                out.append((fn, fa))
        return out

    to, cc = people(em.get("to") or []), people(em.get("cc") or [])
    subject = scrubber.scrub_text(em.get("subject") or "")
    body = scrubber.scrub_text(em.get("body") or "")
    dropped: List[str] = []
    kept: List[Dict[str, Any]] = []
    for att in em.get("attachments") or []:
        name = scrubber.scrub_filename(att.get("name") or "attachment")
        if keep_attachments and att.get("data") is not None:
            kept.append({"name": name, "content_type": att.get("content_type") or "", "data": att["data"],
                         "real_name": att.get("name")})
        else:
            dropped.append(name)
    domain = f_email.rpartition("@")[2] or "scrubbed.invalid"
    mid = f"scrub.{email_id(em)}@{domain}"
    notes: List[str] = []
    if dashes:
        notes.append(f"{dashes} em/en dash(es) turned into plain hyphens (repo rule)")
    date = em.get("date")
    if not parse_iso(date):
        # tests/test_import.py wants a date on every fixture; a fixed one keeps reruns identical.
        date = (datetime(2026, 1, 5, 9, 0) + timedelta(days=_h("date", email_id(em)) % 180)).replace(
            tzinfo=timezone.utc).isoformat()
        notes.append(f"the original has no date, so the file says {date[:10]}")
    if not body.strip():
        notes.append("the body is empty (tests/test_import.py expects text in every body)")
    eml = build_eml(from_name=f_name, from_email=f_email, to=to, cc=cc, subject=subject, date=date,
                    message_id=mid, body=body, dropped=dropped, kept=kept, boundary=f"scrub-{email_id(em)}")
    fields = [("from", f"{f_name} <{f_email}>"), ("to", "; ".join(f"{n} <{a}>" for n, a in to)),
              ("cc", "; ".join(f"{n} <{a}>" for n, a in cc)), ("subject", subject), ("body", body)]
    fields += [("attachment name", n) for n in dropped + [k["name"] for k in kept]]
    parts: Counter = Counter()
    for t in [em.get("subject") or "", em.get("body") or ""] + [a.get("name") or "" for a in em.get("attachments") or []]:
        parts.update(scrubber.find_parts(t))
    return {"eml": eml, "subject": subject, "from_email": f_email, "from_name": f_name, "body": body, "parts": parts,
            "dropped": dropped, "kept": kept, "checks": check_these(scrubber, fields),
            "warnings": list(em.get("warnings") or []), "notes": notes}


def format_report(src: str, target: Path, status: str, res: Dict[str, Any], scrubber: Scrubber,
                  args: argparse.Namespace, is_rfq: bool, roundtrip: List[str]) -> str:
    lines = [f"== {src}",
             f"   {status} {_rel(target)}  (lane {args.lane}, {'RFQ' if is_rfq else 'not an RFQ'})",
             f"   subject: {res['subject']}"]
    if scrubber.stats:
        lines.append("   Replacements:")
        order = ["--map", "person", "name", "email", "url", "domain", "company", "phone", "street",
                 "suite", "po box", "city", "part number"]
        for (kind, real, fake), n in sorted(scrubber.stats.items(),
                                            key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 99,
                                                            kv[0][1].lower())):
            lines.append(f"      {kind:<11} {real!s} -> {fake}  (x{n})")
    else:
        lines.append("   Replacements: none")
    parts = res["parts"]
    if parts:
        verb = "replaced" if scrubber.part_numbers else "KEPT (use --part-numbers to replace)"
        lines.append(f"   Part numbers {verb}:")
        for p, n in sorted(parts.items()):
            tag = "labeled" if p.upper() in scrubber.part_labeled else "looks like one"
            lines.append(f"      {p}  (x{n}, {tag})")
    if res["dropped"]:
        lines.append("   Attachments dropped (listed in X-Scrubbed-Attachments): " + "; ".join(res["dropped"]))
    if res["kept"]:
        lines.append("   !!! Attachments KEPT UNSCRUBBED, check each by hand: " + "; ".join(k["name"] for k in res["kept"]))
    for w in res["warnings"]:
        lines.append(f"   note from the reader: {w}")
    for n in res["notes"]:
        lines.append(f"   note: {n}")
    if roundtrip:
        lines.append("   !!! reading the file back gave a different " + ", ".join(roundtrip))
    if res["checks"]:
        lines.append("   Check these:")
        for c in res["checks"]:
            lines.append(f"      - {c}")
    else:
        lines.append("   Check these: nothing found (still read the file)")
    return "\n".join(lines) + "\n"


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
    sys.exit(run())


if __name__ == "__main__":
    main()
