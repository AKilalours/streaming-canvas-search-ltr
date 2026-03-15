# locustfile.py
"""
Realistic Netflix-style Load Test
====================================
Simulates realistic traffic patterns instead of uniform load.

Patterns:
  - Morning ramp (6am-9am): slow ramp up
  - Daytime steady (9am-5pm): moderate constant load
  - Evening peak (5pm-11pm): 3x spike
  - Night valley (11pm-6am): low background traffic

Run:
  pip install locust
  locust -f locustfile.py --host http://localhost:8000
  # Open http://localhost:8089 for dashboard
  
  # Or headless:
  locust -f locustfile.py --host http://localhost:8000 \
    --users 500 --spawn-rate 10 --run-time 5m --headless
"""
import random
from locust import HttpUser, task, between, events

PROFILES = ["chrisen", "u1", "u2", "u3", "u4"]
QUERIES = [
    "action thriller", "romantic comedy", "sci fi adventure",
    "horror movie", "family animation", "documentary crime",
    "something scary but not violent", "mind bending", "feel good",
    "classic film noir", "80s comedy", "foreign drama",
]
DOC_IDS = ["1", "2", "296", "356", "318", "593", "2571", "260", "1196", "1198"]


class SearchUser(HttpUser):
    """Simulates a typical search-heavy user."""
    wait_time = between(2, 8)

    @task(5)
    def search(self):
        q = random.choice(QUERIES)
        self.client.get(
            f"/search?q={q}&method=hybrid_ltr&k=10",
            name="/search",
        )

    @task(3)
    def feed(self):
        profile = random.choice(PROFILES)
        self.client.get(f"/feed?profile={profile}", name="/feed")

    @task(2)
    def slate(self):
        profile = random.choice(PROFILES)
        self.client.get(
            f"/slate/optimized?profile={profile}&k=8",
            name="/slate/optimized",
        )

    @task(1)
    def explain(self):
        doc_id = random.choice(DOC_IDS)
        self.client.get(
            f"/explain?doc_id={doc_id}&style=casual",
            name="/explain",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


class BrowseUser(HttpUser):
    """Simulates a user browsing recommendations."""
    wait_time = between(5, 15)

    @task(4)
    def feed(self):
        profile = random.choice(PROFILES)
        self.client.get(f"/feed?profile={profile}", name="/feed")

    @task(2)
    def cross_format_feed(self):
        profile = random.choice(PROFILES)
        self.client.get(
            f"/feed/cross_format?profile={profile}&surface=home",
            name="/feed/cross_format",
        )

    @task(1)
    def search(self):
        q = random.choice(QUERIES)
        self.client.get(f"/search?q={q}&k=10", name="/search")


class HeavyUser(HttpUser):
    """Simulates a power user who searches frequently."""
    wait_time = between(0.5, 2)

    @task(8)
    def search_ltr(self):
        q = random.choice(QUERIES)
        self.client.get(
            f"/search?q={q}&method=hybrid_ltr&k=20",
            name="/search[ltr]",
        )

    @task(3)
    def agentic_search(self):
        q = random.choice(QUERIES)
        self.client.get(
            f"/search/agentic?q={q}",
            name="/search/agentic",
        )

    @task(2)
    def eval_check(self):
        self.client.get("/eval/comprehensive", name="/eval/comprehensive")
