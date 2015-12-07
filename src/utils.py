import numpy as np
import sys
import re
import fnmatch as fn
import os
import cgi
import random
import json

def CUncomment(content):
    return re.sub(r'/\*.*?\*/', '', content,
                  flags=re.MULTILINE|re.DOTALL)

def CPPUncomment(content):
    pass1 = CUncomment(content)
    pass2 = re.sub(r'//.*', '', pass1)
    return pass2

def PyUncomment(content):
    """ we want to get rid of comments and docstrings """
    pass1 = re.sub(r'#.*', '', content)
    pass2 = re.sub(r'""".*?"""', '', pass1,
                   flags=re.MULTILINE|re.DOTALL)
    return pass2

def uncomment(fileName, content):
    if fileName.endswith('.c'):
        return CUncomment(content)
    elif (fileName.endswith('.cpp')
          or fileName.endswith('.cc') 
          or fileName.endswith('.cxx')
          or fileName.endswith('.h')
          or fileName.endswith('.hh')
          or fileName.endswith('.hxx')
          or fileName.endswith('.hpp')):
        return CPPUncomment(content)
    elif fileName.endswith('.py'):
        return PyUncomment(content)
    return content

def matchingFiles(data_dirs, suffixes):
    globs = map(lambda x : '*.' + x, suffixes)
    matches = []  # list of files to read
    for data_dir in data_dirs:
        for root, dirnames, filenames in os.walk(data_dir):
            for glob in globs:
                for filename in fn.filter(filenames, glob):
                    matches.append(os.path.join(root, filename))
    return matches

def tokenize(fileName, retcontent=False):
    allTokens = []
    with open(fileName) as data:
        content = uncomment(fileName, data.read())
        for line in content.split('\n'):
            allTokens += [token.strip()
                          for token in re.split(r'(\W+)', line)
                          if len(token.strip()) > 0]
    if not retcontent:
        return allTokens
    else:
        return allTokens, content

def matchTokensToContent(tokens, content):
    cid = 0
    spans = []
    for tid,tok in enumerate(tokens):
        while content[cid] != tok[0]:
            cid += 1
        spans.append((cid, cid+len(tok)))
        cid += len(tok)
    return spans

def colorPrint(*args, **kwargs):
    cmap = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'black': '\033[30m',
    }
    color = kwargs['color'] if 'color' in kwargs else 'black'
    if sys.stdout.isatty():
        sys.stdout.write(cmap[color])
        for ii,arg in enumerate(args):
            sys.stdout.write(str(arg))
            if ii != len(args)-1:
                sys.stdout.write(' ')
        sys.stdout.write('\033[00m')
    else:
        for arg in args:
            sys.stdout.write(str(arg))


def HTMLHeader():
    return """<!DOCTYPE html>
<html>
<head>
<style>
pre {
    color: #888888;
}
span.c {
    background-color: #CCFFCC;
}
span.pc {
    background-color: #FFEEBB;
}
span.w {
    background-color: #FFCCCC;
}
span.tok {
    color: black;
}
div.popup {
    color: black;
    background-color: #EFEFFF;
    outline: 2px solid blue;
    position: fixed;
    left: 50px;
    top: 50px;
    display: none;
    z-index: 10;
    padding: 2px 2px 2px 2px;
}
span.tok {
    position: relative;
    z-index: 1;
}
</style>
<script src="https://code.jquery.com/jquery-1.10.2.js"></script>
</head>
<body>
<pre>
<div id="pdiv" class="popup">blah<br/>blah</div>
"""

def attn_to_json(attn):
    return '[' + ','.join(['%.3f' % x for x in attn]) + ']'

def attns_to_json(attns):
    return '[' + ','.join([attn_to_json(attn) for attn in attns]) + ']'

def HTMLFooter(candidates, attentions):
    return """</pre>
<script>
    var candidates = """ + json.dumps(candidates) + """;
    var attentions = """ + attns_to_json(attentions) + """;
    $("span.tok")
    .mouseenter(function() {
        var myID = parseInt($(this).attr("id"));
        for (nbrID = myID - 40; nbrID < myID; nbrID++) {
            if(nbrID > 0) {
                var nbrIDStr = nbrID.toString();
                document.getElementById(nbrIDStr).style.fontWeight = "bold";
                var allnbrs = document.getElementsByClassName("t" + nbrIDStr);
                var d = 40 - (myID - nbrID);
                var attn = attentions[myID-1][d];
                var r = Math.floor(102 + (255-102)*(1-attn));
                for(i = 0; i < allnbrs.length; i++) {
                    allnbrs[i].style.backgroundColor = "rgb(" + r + ",255,255)";
                }
                document.getElementById(nbrIDStr).style.color = "black";
            }
        }
        $(this).css("outline", "2px dotted red");
        $(this).css("font-weight", "bold");
        $(this).css("color", "black");
        $(this).css("z-index", "2");

        var rect = $(this)[0].getBoundingClientRect();
        var popup = document.getElementById("pdiv");
        popup.style.display = "inline";
        popup.style.left = rect.left+"px";
        popup.style.top = (rect.bottom+2)+"px";
        popup.innerHTML = candidates[myID-1].join('<br/>');
    })
    .mouseleave(function() {
        var myID = parseInt($(this).attr("id"));
        for (nbrID = myID - 40; nbrID < myID; nbrID++) {
            if(nbrID > 0) {
                var nbrIDStr = nbrID.toString();
                document.getElementById(nbrIDStr).style.fontWeight = "normal";
                var allnbrs = document.getElementsByClassName("t" + nbrIDStr);
                for(i = 0; i < allnbrs.length; i++) {
                    allnbrs[i].style.backgroundColor = "inherit";
                }
                document.getElementById(nbrIDStr).style.color = "black";
            }
        }
        $(this).css("outline", "none");
        $(this).css("font-weight", "normal");
        $(this).css("color", "black");
        $(this).css("z-index", "1");

        var popup = document.getElementById("pdiv");
        popup.style.display = "none";
    });
</script>
</body>
</html>
"""

def spanStart(annotation, sid):
    annClass = {0: 'w', 1: 'pc', 2: 'c'}
    if annotation in annClass:
        annspan = "<span class=\"%s\">" % annClass[annotation]
        tokspan = "<span class=\"t%d\">" % sid
        return annspan + tokspan
    else:
        return ""

def spanEnd(annotation):
    annClass = {0: 'w', 1: 'pc', 2: 'c'}
    if annotation in annClass:
        return "</span></span>"
    else:
        return ""

def colorizedHTML(content, annotations, spans):
    """
    annotation convention:
    0 -> wrong
    1 -> partially correct
    2 -> correct
    -1 -> no annotation
    """
    pa = None
    s = ''
    sid = 0
    lno = 1
    for i,(c,a) in enumerate(zip(content, annotations)):
        if sid < len(spans) and spans[sid][0] == i:
            # start a new token span
            s += "<span class=\"tok\" id=\"%d\">" % (sid+1)
            # and a new coloring span
            s += spanStart(a, sid+1)
        elif sid < len(spans) and a != pa:
#        if a != pa:
            # start a new coloring span
            s += spanEnd(pa)
            s += spanStart(a, sid+1)
        s += cgi.escape(c)
        if c == '\n':
            s += '%4d    ' % lno
            lno += 1
        pa = a
        if sid < len(spans) and spans[sid][1]-1 == i:
            # end the coloring span
            s += "</span></span>"
            # and the token span
            s += "</span>"
            sid += 1
    s += spanEnd(pa)
    return s


def softmaxLossAndGrads(scores, tgts):
    deltas = scores - np.amax(scores, axis=1, keepdims=True)
    edeltas = np.exp(deltas)
    probs = edeltas/np.sum(edeltas, axis=1, keepdims=True)
    grads = probs.copy()
    nb = scores.shape[0]
    grads[np.arange(nb),tgts] -= 1
    loss = -np.mean(np.log(probs[np.arange(nb),tgts]+1e-13))
    return loss, probs, grads

def relu(x):
    return 0.5*(x + np.abs(x))

def drelu(dy, y):
    return dy*(y > 0)

class WeightedRandomSampler:
    def __init__(self, weights):
        """
        weights should be integer array
        """
        N = len(weights)
        S = sum(weights)
        self.H = S // N
        # normalize the weights array so that its sum is exactly H*N
        self.weights = [w for w in weights]
        residue = S - self.H*N
        for j in range(residue):
            self.weights[j] -= 1
        # ok now let us generate the alias array
        overfull = [i for i,w in enumerate(self.weights)
                    if w > self.H]
        underfull = [i for i,w in enumerate(self.weights)
                     if w < self.H]
        self.K = [-1 for _ in range(N)]
        self.U = [self.H for _ in range(N)]

        while len(underfull) > 0:
            # get one from overfull and one from underfull
            ov = overfull.pop()
            un = underfull.pop()
            # set the alias entry for un
            self.K[un] = ov
            self.U[un] = self.weights[un]
            # compute correct value for overfull
            self.weights[ov] = self.weights[ov] + self.weights[un] - self.H
            # make un exact
            self.weights[un] = self.H
            if self.weights[ov] > self.H:
                overfull.append(ov)
            elif self.weights[ov] < self.H:
                underfull.append(ov)

    def generate(self):
        idx = np.random.randint(len(self.weights))
        r = np.random.randint(self.H+1)
        if r <= self.U[idx]:
            return idx
        else:
            return self.K[idx]

if __name__ == '__main__':
    sampler = WeightedRandomSampler([10,3,5,4,7,9,2])
    for _ in range(20):
        print sampler.generate()
