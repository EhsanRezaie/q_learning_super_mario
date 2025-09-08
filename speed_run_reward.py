import gym

class SpeedrunReward(gym.Wrapper):
    def __init__(self, env, stuck_limit=60):
        """
        stuck_limit: تعداد فریم‌هایی که ماریو می‌تونه ثابت بمونه قبل از جریمه
        """
        super().__init__(env)
        self.prev_x = 0
        self.prev_time = None
        self.stuck_counter = 0
        self.stuck_limit = stuck_limit

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.prev_x = 0
        self.prev_time = None
        self.stuck_counter = 0
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # جلو رفتن
        curr_x = info['x_pos']
        delta_x = curr_x - self.prev_x

        # بررسی گیر کردن
        if delta_x == 0:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.prev_x = curr_x

        # زمان سپری شده
        if self.prev_time is None:
            self.prev_time = info['time']
        delta_time = self.prev_time - info['time']
        self.prev_time = info['time']

        # جریمه مردن یا گیر کردن
        death_penalty = -50 if done else 0
        stuck_penalty = 0
        if self.stuck_counter >= self.stuck_limit:
            stuck_penalty = -50
            done = True  # اپیزود تموم میشه

        # reward نهایی
        new_reward = (delta_x * 0.1) - (delta_time * 0.5) + death_penalty + stuck_penalty

        return state, new_reward, done, info
