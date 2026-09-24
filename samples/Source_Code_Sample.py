import json
import random
import time
import random
import threading

class DecayProxyRotator:
    def __init__(self, proxy_list):
        self.lock = threading.Lock()
        self.proxy_pool = {
            p: {
                "score": 100.0,
                "last_used": time.time(),
                "base_score": 100.0,
                "failure_count": 0,
                "penalty_factor": 1.0  # Multiplier for recovery time
            } for p in proxy_list
        }
        self.usage_cost = 25
        self.recovery_rate = 0.05
        # Higher values make it harder for failing proxies to return
        self.penalty_increment = 2.0

    def _calculate_current_score(self, stats):
        """Calculates the score lazily based on elapsed time and penalties."""
        now = time.time()
        elapsed = now - stats["last_used"]
        
        # Recovery is slowed down by the penalty_factor
        # Effective recovery rate = base_rate / penalty_factor
        effective_recovery = (elapsed * self.recovery_rate) / stats["penalty_factor"]
        
        current_score = min(stats["base_score"], stats["score"] + effective_recovery)
        return max(0, current_score)

    def get_proxy(self):
        with self.lock:
            proxies_keys = list(self.proxy_pool.keys())
            weights = []
            
            for p in proxies_keys:
                # Update score based on time passed since it was last touched
                current_score = self._calculate_current_score(self.proxy_pool[p])
                self.proxy_pool[p]["score"] = current_score
                weights.append(current_score)

            if sum(weights) == 0:
                raise Exception("All proxies are cooling down or penalized.")

            selected = random.choices(proxies_keys, weights=weights, k=1)[0]
            
            # Deduct score for usage
            self.proxy_pool[selected]["score"] -= self.usage_cost
            self.proxy_pool[selected]["last_used"] = time.time()
            return selected

    def report_failure(self, proxy):
        """Increase penalty multiplier and reset score to zero."""
        with self.lock:
            if proxy in self.proxy_pool:
                stats = self.proxy_pool[proxy]
                stats["score"] = 0
                stats["failure_count"] += 1
                # Increase the penalty factor (linear or exponential)
                # A factor of 2.0 means it recovers 2x slower than normal.
                stats["penalty_factor"] += self.penalty_increment
                stats["last_used"] = time.time()

    def report_success(self, proxy):
        """Gradually reduce the penalty on successful requests."""
        with self.lock:
            if proxy in self.proxy_pool:
                # Slowly bring the penalty factor back toward 1.0
                current_pf = self.proxy_pool[proxy]["penalty_factor"]
                self.proxy_pool[proxy]["penalty_factor"] = max(1.0, current_pf - 0.1)


class UAFreshnessRotator:
    def __init__(self, ua_list):
        self.lock = threading.Lock()
        self.ua_pool = {
            ua: {
                "score": 100.0,
                "last_used": time.time(),
                "base_score": 100.0,
                "block_count": 0,
                "backoff_multiplier": 1.0 
            } for ua in ua_list
        }
        # Cost of "burning" a UA. 
        # Higher cost = more diversity (forces rotation through the whole list)
        self.usage_burn = 25 
        self.recovery_rate = 0.05  # Points recovered per second
        self.penalty_increment = 2.0

    def _get_current_freshness(self, stats):
        now = time.time()
        elapsed = now - stats["last_used"]
        
        # Recovery is slowed if the UA has been "blocked" recently
        recovery = (elapsed * self.recovery_rate) / stats["backoff_multiplier"]
        
        freshness = min(stats["base_score"], stats["score"] + recovery)
        return max(0, freshness)

    def get_ua(self):
        with self.lock:
            ua_keys = list(self.ua_pool.keys())
            weights = []
            
            for ua in ua_keys:
                freshness = self._get_current_freshness(self.ua_pool[ua])
                self.ua_pool[ua]["score"] = freshness
                weights.append(freshness)

            if sum(weights) == 0:
                # Fallback: if all are burned, pick purely at random or wait
                return random.choice(ua_keys)

            selected = random.choices(ua_keys, weights=weights, k=1)[0]
            
            # Burn some freshness so we don't reuse it immediately
            self.ua_pool[selected]["score"] -= self.usage_burn
            self.ua_pool[selected]["last_used"] = time.time()
            return selected

    def report_block(self, ua):
        """Called when a UA hits a CAPTCHA or 403."""
        with self.lock:
            if ua in self.ua_pool:
                stats = self.ua_pool[ua]
                stats["score"] = 0  # Force immediate cooldown
                stats["block_count"] += 1
                # Increase backoff: flagged UAs stay 'stale' for much longer
                stats["backoff_multiplier"] += self.penalty_increment
                stats["last_used"] = time.time()


# --- TEST SCRIPT ---

def run_test():
    with open(r'E:\Dev\Future\badger\api\proxyList.json', 'r') as f:
        proxies = json.load(f)

    rotator = DecayProxyRotator(proxies)
    
    print(f"Simulating 100 requests with random failures...")
    
    for i in range(1, 101):
        proxy = rotator.get_proxy()
        
        # SIMULATION: Randomly fail 10% of the time to test the report
        if random.random() < 0.10:
            rotator.report_failure(proxy)
        
        time.sleep(0.01) 

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("TEST REPORT")
    print("="*50)
    
    used_proxies = [p for p, s in rotator.proxy_pool.items() if s["usage_count"] > 0]
    print(f"Unique proxies used: {len(used_proxies)} / {len(proxies)}")

    # 1. Top Used Report
    print("\n[ TOP 5 MOST USED PROXIES ]")
    sorted_usage = sorted(rotator.proxy_pool.items(), key=lambda x: x[1]['usage_count'], reverse=True)
    for p, stats in sorted_usage[:5]:
        print(f"Proxy: {p[:25]}... | Used: {stats['usage_count']} | Score: {stats['score']:.1f}")

    # 2. Top Failed Report
    print("\n[ TOP 5 MOST FAILED PROXIES ]")
    # Sort by failure_count
    sorted_failures = sorted(rotator.proxy_pool.items(), key=lambda x: x[1]['failure_count'], reverse=True)
    
    # Filter only those that actually failed at least once
    failed_proxies = [item for item in sorted_failures if item[1]['failure_count'] > 0]
    
    if not failed_proxies:
        print("No failures recorded during this run.")
    else:
        for p, stats in failed_proxies[:5]:
            # Calculate failure rate for better context
            fail_rate = (stats['failure_count'] / stats['usage_count']) * 100
            print(f"Proxy: {p[:25]}... | Fails: {stats['failure_count']} | Rate: {fail_rate:.1f}% | Score: {stats['score']:.1f}")
    
    print("="*50)

if __name__ == "__main__":
    run_test()